import { describe, expect, it } from 'vitest';
import {
  collectFromDrop,
  collectFromInput,
  isAcceptedFile,
  makeIdentifier,
} from './fileSource';

function fakeFile(name: string, webkitRelativePath = ''): File {
  const f = new File(['x'], name, { type: 'image/jpeg' });
  if (webkitRelativePath) {
    Object.defineProperty(f, 'webkitRelativePath', { value: webkitRelativePath });
  }
  return f;
}

function fakeFileList(files: File[]): FileList {
  return {
    length: files.length,
    item: (i: number) => files[i] ?? null,
    [Symbol.iterator]: files[Symbol.iterator].bind(files),
  } as unknown as FileList;
}

describe('collectFromInput', () => {
  it('uses webkitRelativePath when present, else the bare file name', () => {
    const f1 = fakeFile('a.jpg', 'folder/sub/a.jpg');
    const f2 = fakeFile('b.jpg');
    const result = collectFromInput(fakeFileList([f1, f2]));
    expect(result.map((r) => r.relPath)).toEqual(['folder/sub/a.jpg', 'b.jpg']);
    expect(result[0]!.id).toBe('folder/sub/a.jpg');
  });
});

describe('isAcceptedFile', () => {
  it('is case-insensitive', () => {
    expect(isAcceptedFile('IMG.JPG', ['.jpg', '.png'])).toBe(true);
    expect(isAcceptedFile('img.Png', ['.jpg', '.png'])).toBe(true);
    expect(isAcceptedFile('img.gif', ['.jpg', '.png'])).toBe(false);
  });
});

describe('makeIdentifier', () => {
  it('normalizes backslashes and strips a leading ./ or /', () => {
    expect(makeIdentifier('src/', 'a\\b\\c.jpg')).toBe('src/a/b/c.jpg');
    expect(makeIdentifier('src/', './a/b.jpg')).toBe('src/a/b.jpg');
    expect(makeIdentifier('src/', '/a/b.jpg')).toBe('src/a/b.jpg');
  });

  it('rejects a .. segment', () => {
    expect(makeIdentifier('src/', '../../etc/passwd')).toBeNull();
    expect(makeIdentifier('src/', 'a/../b.jpg')).toBeNull();
  });
});

describe('collectFromDrop', () => {
  interface FakeEntry {
    isFile: boolean;
    isDirectory: boolean;
    fullPath: string;
    name: string;
    file?(success: (f: File) => void, error: (e: unknown) => void): void;
    createReader?(): {
      readEntries(
        success: (entries: FakeEntry[]) => void,
        error: (e: unknown) => void,
      ): void;
    };
  }

  function fileEntry(fullPath: string): FakeEntry {
    return {
      isFile: true,
      isDirectory: false,
      fullPath,
      name: fullPath.split('/').pop()!,
      file: (success) => success(fakeFile(fullPath.split('/').pop()!)),
    };
  }

  function dirEntry(fullPath: string, children: FakeEntry[]): FakeEntry {
    // Simulate the real API's "at most ~2 per call" pagination behavior
    // so the readEntries-loop test actually proves the loop, not a
    // single lucky call.
    let index = 0;
    const BATCH = 2;
    return {
      isFile: false,
      isDirectory: true,
      fullPath,
      name: fullPath.split('/').pop()!,
      createReader: () => ({
        readEntries: (success) => {
          const batch = children.slice(index, index + BATCH);
          index += BATCH;
          success(batch);
        },
      }),
    };
  }

  function itemsFor(entries: FakeEntry[]): DataTransferItemList {
    const items = entries.map((e) => ({
      webkitGetAsEntry: () => e,
    }));
    return items as unknown as DataTransferItemList;
  }

  it('walks a nested directory and drains more than one readEntries batch', async () => {
    const many = Array.from({ length: 7 }, (_, i) => fileEntry(`drop/sub/f${i}.jpg`));
    const root = dirEntry('drop', [dirEntry('drop/sub', many)]);
    const out: string[] = [];
    for await (const f of collectFromDrop(itemsFor([root]))) {
      out.push(f.relPath);
    }
    expect(out.sort()).toEqual(many.map((e) => e.fullPath).sort());
  });

  it('rejects a .. path from a malicious entry.fullPath', async () => {
    const bad = fileEntry('drop/../../etc/passwd');
    const out: string[] = [];
    for await (const f of collectFromDrop(itemsFor([bad]))) {
      out.push(f.relPath);
    }
    expect(out).toEqual([]);
  });
});
