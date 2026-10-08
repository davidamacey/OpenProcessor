/**
 * F-63(a) (fresh-start findings 2026-09-25): the trainer's log is a
 * terminal stream — every progress line starts with an ANSI erase-line
 * (`ESC[K`) and may carry colors or `\r`-overwritten progress updates,
 * which the `/train` log tail rendered as boxed glyphs. Display only: the
 * served text is otherwise untouched.
 */

// CSI (`ESC [ … final`) and OSC (`ESC ] … BEL|ST`) sequences, plus any
// other two-byte `ESC x` escape.
// eslint-disable-next-line no-control-regex
const ANSI = /\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]/g;

/** What a terminal would show for one served log line. */
export function cleanLogLine(line: string): string {
  const noAnsi = line.replace(ANSI, '');
  // A `\r` rewinds to column 0: the text after the last one is what the
  // terminal ends up showing (progress bars redraw this way).
  const parts = noAnsi.split('\r').filter((p) => p.length > 0);
  return parts.length > 0 ? parts[parts.length - 1]! : '';
}
