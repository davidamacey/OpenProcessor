import React from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Lightbox from '@site/src/components/Lightbox';

/**
 * Renders a screenshot from `static/img/screenshots/<name>` when present,
 * or a clearly-marked "pending" placeholder when it is not.
 *
 * Screenshots may ONLY be captured from an instance holding public sample
 * data (COCO val2017 / Open Images); never a real deployment's imagery.
 * See docs/developer-guide/screenshots.
 *
 * Docusaurus's build-time asset pipeline can't tell us "does this file
 * exist" at MDX-compile time without a webpack loader, so this component
 * resolves the URL and lets the browser's own `onError` flip it to the
 * pending state — cheap, and correct for both `npm start` and `npm run build`.
 *
 * The image is a button that opens it large in a `Lightbox`. Standalone
 * (an MDX doc page) it opens its own single-image lightbox; inside a
 * gallery the parent passes `onOpen` and owns one lightbox for all images
 * so the arrow keys can page between them.
 */
export default function Screenshot({
  name,
  alt,
  caption,
  onOpen,
  width = 1600,
  height = 1000,
}: {
  name: string;
  alt: string;
  caption?: string;
  onOpen?: () => void;
  /** Intrinsic size; reserves the box before the lazy image loads (no layout shift). */
  width?: number;
  height?: number;
}): React.JSX.Element {
  const [failed, setFailed] = React.useState(false);
  const [open, setOpen] = React.useState(false);
  const src = useBaseUrl(`img/screenshots/${name}`);

  if (failed) {
    return (
      <div className="cw-placeholder">
        Screenshot pending — captured from an instance holding public sample data (COCO
        val2017 / Open Images). Expected at <code>static/img/screenshots/{name}</code>.
        {caption ? <div style={{marginTop: '0.5rem', fontStyle: 'italic'}}>{caption}</div> : null}
      </div>
    );
  }

  return (
    <figure style={{margin: '1.5rem 0'}}>
      <button
        type="button"
        className="cw-zoomable"
        aria-label={`Open larger: ${caption ?? alt}`}
        onClick={() => (onOpen ? onOpen() : setOpen(true))}>
        <img
          src={src}
          alt={alt}
          loading="lazy"
          width={width}
          height={height}
          style={{
            width: '100%',
            height: 'auto',
            borderRadius: 8,
            border: '1px solid var(--cw-border)',
            display: 'block',
          }}
          onError={() => setFailed(true)}
        />
      </button>
      {caption ? (
        <figcaption style={{textAlign: 'center', marginTop: '0.5rem', opacity: 0.75}}>
          {caption}
        </figcaption>
      ) : null}
      {open && !onOpen ? (
        <Lightbox images={[{src, alt, caption}]} index={0} onClose={() => setOpen(false)} />
      ) : null}
    </figure>
  );
}
