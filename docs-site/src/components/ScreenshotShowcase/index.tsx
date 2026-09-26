import React from 'react';
import Screenshot from '@site/src/components/Screenshot';
import screenshots from '@site/src/data/screenshots.json';
import styles from './styles.module.css';

/**
 * Data-driven showcase — the route/state list lives in
 * `src/data/screenshots.json`; the capture script's route list is
 * `src/data/screenshot_routes.json`.
 * Each slot degrades to the `<Screenshot>` pending placeholder until the
 * file exists under `static/img/screenshots/`.
 */
export default function ScreenshotShowcase(): React.JSX.Element {
  return (
    <section className={styles.section}>
      <div className="container">
        <h2 className={styles.heading}>See it in action</h2>
        <div className={styles.grid}>
          {screenshots.map((s) => (
            <Screenshot key={s.name} name={s.name} alt={s.alt} caption={s.caption} />
          ))}
        </div>
        <p className={styles.credits}>
          Sample imagery: <a href="https://cocodataset.org">COCO</a> val2017 and{' '}
          <a href="https://storage.googleapis.com/openimages/web/index.html">Open Images</a>{' '}
          (Creative Commons licensed photographs).{' '}
          <a href="docs/developer-guide/screenshots#image-credits">Credits</a>
        </p>
      </div>
    </section>
  );
}
