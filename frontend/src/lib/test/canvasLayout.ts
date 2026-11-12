/**
 * jsdom lays nothing out. This gives an editor canvas a fixed size: the container is `w` by `h`,
 * and the picture frame is wherever its inline style puts it, so pointer maths can be tested.
 */
import { vi } from 'vitest';

function px(el: HTMLElement, name: 'left' | 'top' | 'width' | 'height'): number {
  return parseFloat(el.style.getPropertyValue(name)) || 0;
}

export function stubCanvasLayout(w: number, h: number): void {
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(w);
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(h);
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (
    this: Element,
  ) {
    const el = this as HTMLElement;
    const framed = el.dataset.testid === 'editor-frame';
    const [left, top, width, height] = framed
      ? [px(el, 'left'), px(el, 'top'), px(el, 'width'), px(el, 'height')]
      : [0, 0, w, h];
    return {
      left,
      top,
      width,
      height,
      right: left + width,
      bottom: top + height,
    } as DOMRect;
  });
  HTMLElement.prototype.setPointerCapture = () => {};
}

/** A synthetic pointer event; jsdom has no `PointerEvent`. */
export function pointer(el: Element, type: string, x: number, y: number): void {
  el.dispatchEvent(
    Object.assign(new Event(type, { bubbles: true }), {
      clientX: x,
      clientY: y,
      pointerId: 1,
      button: 0,
    }),
  );
}
