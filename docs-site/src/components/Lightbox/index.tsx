import React, {useCallback, useEffect, useRef} from 'react';
import {createPortal} from 'react-dom';
import styles from './styles.module.css';

export type LightboxImage = {src: string; alt: string; caption?: string};

/**
 * Near-fullscreen image viewer, no dependencies.
 *
 * Rendered into a portal only while open, so it costs nothing on the page
 * and never shifts layout. Esc or a click on the backdrop closes it;
 * Left/Right move through `images` when there is more than one. Focus
 * moves to the close button on open and returns to the opener on close,
 * and Tab stays inside the dialog. Generic: a docs site passes any list of
 * images; `Screenshot` and `ScreenshotShowcase` are the two callers here.
 */
export default function Lightbox({
  images,
  index,
  onClose,
  onIndexChange,
}: {
  images: LightboxImage[];
  index: number;
  onClose: () => void;
  onIndexChange?: (next: number) => void;
}): React.JSX.Element | null {
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const count = images.length;
  const image = images[index];
  const canPage = count > 1 && Boolean(onIndexChange);

  const go = useCallback(
    (delta: number) => {
      if (canPage) onIndexChange?.((index + delta + count) % count);
    },
    [canPage, count, index, onIndexChange],
  );

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();

    // Lock page scroll without the page jumping when the scrollbar goes away.
    const root = document.documentElement;
    const scrollbar = window.innerWidth - root.clientWidth;
    const prevOverflow = root.style.overflow;
    const prevPadding = root.style.paddingRight;
    root.style.overflow = 'hidden';
    if (scrollbar > 0) root.style.paddingRight = `${scrollbar}px`;

    return () => {
      root.style.overflow = prevOverflow;
      root.style.paddingRight = prevPadding;
      opener?.focus?.();
    };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        go(1);
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        go(-1);
      } else if (e.key === 'Tab' && dialogRef.current) {
        const focusable = dialogRef.current.querySelectorAll<HTMLElement>('button');
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [go, onClose]);

  if (!image || typeof document === 'undefined') return null;

  return createPortal(
    <div
      className={styles.backdrop}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}>
      <div
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-label={image.caption ?? image.alt}
        onClick={(e) => {
          if (e.target === e.currentTarget) onClose();
        }}>
        <button
          ref={closeRef}
          type="button"
          className={styles.close}
          aria-label="Close image viewer"
          onClick={onClose}>
          ×
        </button>
        {canPage && (
          <button
            type="button"
            className={`${styles.nav} ${styles.prev}`}
            aria-label="Previous image"
            onClick={() => go(-1)}>
            ‹
          </button>
        )}
        <figure className={styles.figure}>
          <img className={styles.image} src={image.src} alt={image.alt} />
          <figcaption className={styles.caption}>
            {image.caption}
            {canPage && (
              <span className={styles.counter}>
                {' '}
                ({index + 1} / {count})
              </span>
            )}
          </figcaption>
        </figure>
        {canPage && (
          <button
            type="button"
            className={`${styles.nav} ${styles.next}`}
            aria-label="Next image"
            onClick={() => go(1)}>
            ›
          </button>
        )}
      </div>
    </div>,
    document.body,
  );
}
