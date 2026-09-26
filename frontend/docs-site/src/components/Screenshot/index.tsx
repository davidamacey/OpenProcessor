import React from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';

/**
 * Renders a screenshot from `static/img/screenshots/<name>` when present,
 * or a clearly-marked "pending" placeholder when it is not.
 *
 * Cropwright's screenshots may ONLY be captured from an instance pointed at
 * a backend holding public sample data (COCO val2017 / Open Images plates —
 * see `scripts/capture_docs_screenshots.py` and
 * docs-site/docs/developer-guide/screenshots.md). Never a real deployment's
 * imagery. Until that capture run happens, every doc page that wants an
 * image renders this pending block instead of a broken <img> or, worse, a
 * placeholder that looks like a real screenshot.
 *
 * Docusaurus's build-time asset pipeline can't tell us "does this file
 * exist" at MDX-compile time without a webpack loader, so this component
 * resolves the URL and lets the browser's own `onError` flip it to the
 * pending state — cheap, and correct for both `npm start` and `npm run build`.
 */
export default function Screenshot({
  name,
  alt,
  caption,
}: {
  name: string;
  alt: string;
  caption?: string;
}): React.JSX.Element {
  const [failed, setFailed] = React.useState(false);
  const src = useBaseUrl(`img/screenshots/${name}`);

  if (failed) {
    return (
      <div className="cw-placeholder">
        Screenshot pending — captured from a Cropwright instance pointed at a public-data
        OpenProcessor backend (COCO val2017 / Open Images plates). Expected at{' '}
        <code>static/img/screenshots/{name}</code>.
        {caption ? <div style={{marginTop: '0.5rem', fontStyle: 'italic'}}>{caption}</div> : null}
      </div>
    );
  }

  return (
    <figure style={{margin: '1.5rem 0'}}>
      <img
        src={src}
        alt={alt}
        loading="lazy"
        style={{
          width: '100%',
          borderRadius: 8,
          border: '1px solid var(--cw-border)',
          display: 'block',
        }}
        onError={() => setFailed(true)}
      />
      {caption ? (
        <figcaption style={{textAlign: 'center', marginTop: '0.5rem', opacity: 0.75}}>
          {caption}
        </figcaption>
      ) : null}
    </figure>
  );
}
