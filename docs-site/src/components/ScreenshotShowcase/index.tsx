import React from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Screenshot from '@site/src/components/Screenshot';
import Lightbox from '@site/src/components/Lightbox';
import screenshots from '@site/src/data/screenshots.json';
import styles from './styles.module.css';

/**
 * Data-driven showcase — the route/state list lives in
 * `src/data/screenshots.json`; the capture script's route list is
 * `src/data/screenshot_routes.json`.
 * Each slot degrades to the `<Screenshot>` pending placeholder until the
 * file exists under `static/img/screenshots/`. One shared `Lightbox` pages
 * through every image in the gallery with the arrow keys.
 */
export default function ScreenshotShowcase(): React.JSX.Element {
  const [openIndex, setOpenIndex] = React.useState<number | null>(null);
  const base = useBaseUrl('img/screenshots/');
  const images = screenshots.map((s) => ({src: `${base}${s.name}`, alt: s.alt, caption: s.caption}));

  return (
    <section className={styles.section}>
      <div className="container">
        <h2 className={styles.heading}>See it in action</h2>
        <div className={styles.grid}>
          {screenshots.map((s, i) => (
            <Screenshot
              key={s.name}
              name={s.name}
              alt={s.alt}
              caption={s.caption}
              onOpen={() => setOpenIndex(i)}
            />
          ))}
        </div>
        <p className={styles.credits}>
          Sample imagery: <a href="https://cocodataset.org">COCO</a> val2017 and{' '}
          <a href="https://storage.googleapis.com/openimages/web/index.html">Open Images</a>{' '}
          (Creative Commons licensed photographs).{' '}
          <a href="docs/developer-guide/screenshots#image-credits">Credits</a>
        </p>
      </div>
      {openIndex !== null ? (
        <Lightbox
          images={images}
          index={openIndex}
          onClose={() => setOpenIndex(null)}
          onIndexChange={setOpenIndex}
        />
      ) : null}
    </section>
  );
}
