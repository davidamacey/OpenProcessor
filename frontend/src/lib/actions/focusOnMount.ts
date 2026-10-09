/**
 * Move focus to the node once it mounts.
 *
 * Modal shells need this: their Escape-to-dismiss handler lives on the
 * backdrop, so until something inside the dialog holds focus the keystroke
 * never reaches it and the only way out is the mouse. Focusing the shell
 * also puts a screen reader inside the dialog rather than leaving it on
 * whatever button opened it.
 *
 * `select: true` additionally selects the text of an input, for the
 * click-to-rename affordances where the operator types over the old value.
 */
export function focusOnMount(node: HTMLElement, opts: { select?: boolean } = {}): void {
  queueMicrotask(() => {
    node.focus();
    if (opts.select && node instanceof HTMLInputElement) node.select();
  });
}
