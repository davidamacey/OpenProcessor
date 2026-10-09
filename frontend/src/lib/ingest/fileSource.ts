/**
 * Turns a file-picker `FileList` or a drag-drop `DataTransferItemList`
 * into the flat `IngestFile[]` the rest of `$lib/ingest` works with.
 * docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §A.2.
 */

export interface IngestFile {
  /** Stable within one selection: `relPath` (unique per selection by
   *  construction — the browser/filesystem won't hand back two entries
   *  at the same relative path). */
  id: string;
  relPath: string;
  file: File;
  size: number;
}

/** `file.webkitRelativePath` is populated only for a `webkitdirectory`
 *  input selection; a plain multi-file picker leaves it `''`. */
export function collectFromInput(files: FileList): IngestFile[] {
  const out: IngestFile[] = [];
  for (const file of Array.from(files)) {
    const relPath = normalizeRelPath(file.webkitRelativePath || file.name);
    if (hasDotDotSegment(relPath)) continue;
    out.push({ id: relPath, relPath, file, size: file.size });
  }
  return out;
}

export function isAcceptedFile(name: string, exts: string[]): boolean {
  const lower = name.toLowerCase();
  return exts.some((ext) => lower.endsWith(ext.toLowerCase()));
}

/** Normalizes a relative path: backslashes to `/`, strips a leading `./`
 *  or `/`. Both collectors skip any path with a `..` segment, so no
 *  `IngestFile` carries a directory-traversal identifier. */
function normalizeRelPath(raw: string): string {
  let p = raw.replace(/\\/g, '/');
  while (p.startsWith('./')) p = p.slice(2);
  while (p.startsWith('/')) p = p.slice(1);
  return p;
}

function hasDotDotSegment(relPath: string): boolean {
  return relPath.split('/').some((seg) => seg === '..');
}

interface FileSystemEntryLike {
  isFile: boolean;
  isDirectory: boolean;
  fullPath: string;
  name: string;
  file?(success: (f: File) => void, error: (e: unknown) => void): void;
  createReader?(): {
    readEntries(
      success: (entries: FileSystemEntryLike[]) => void,
      error: (e: unknown) => void,
    ): void;
  };
}

function readEntryFile(entry: FileSystemEntryLike): Promise<File> {
  return new Promise((resolve, reject) => {
    if (!entry.file) {
      reject(new Error(`not a file entry: ${entry.fullPath}`));
      return;
    }
    entry.file(resolve, reject);
  });
}

/**
 * `readEntries()` returns at most ~100 entries per call and must be
 * called repeatedly until it returns an empty array — a single call is
 * a common, silent truncation bug for a large drop.
 */
async function drainDirectory(
  entry: FileSystemEntryLike,
): Promise<FileSystemEntryLike[]> {
  const reader = entry.createReader?.();
  if (!reader) return [];
  const all: FileSystemEntryLike[] = [];
  for (;;) {
    const batch = await new Promise<FileSystemEntryLike[]>((resolve, reject) => {
      reader.readEntries(resolve, reject);
    });
    if (batch.length === 0) break;
    all.push(...batch);
  }
  return all;
}

async function* walkEntry(entry: FileSystemEntryLike): AsyncIterable<IngestFile> {
  if (entry.isFile) {
    const relPath = normalizeRelPath(entry.fullPath);
    if (hasDotDotSegment(relPath)) return;
    const file = await readEntryFile(entry);
    yield { id: relPath, relPath, file, size: file.size };
    return;
  }
  if (entry.isDirectory) {
    for (const child of await drainDirectory(entry)) {
      yield* walkEntry(child);
    }
  }
}

/**
 * Streams every file entry out of a drag-drop's
 * `DataTransferItemList`, walking directories depth-first via
 * `webkitGetAsEntry()`.
 */
export async function* collectFromDrop(
  items: DataTransferItemList,
): AsyncIterable<IngestFile> {
  for (const item of Array.from(items)) {
    const asAny = item as unknown as { webkitGetAsEntry?(): FileSystemEntryLike | null };
    const entry = asAny.webkitGetAsEntry?.();
    if (entry) {
      yield* walkEntry(entry);
      continue;
    }
    // No filesystem-entry API (non-Chromium browser): fall back to the
    // plain File the drop still carries.
    const file = item.getAsFile?.();
    if (file) {
      const relPath = normalizeRelPath(file.name);
      yield { id: relPath, relPath, file, size: file.size };
    }
  }
}
