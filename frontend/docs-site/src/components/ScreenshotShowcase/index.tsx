import React from 'react';
import Screenshot from '@site/src/components/Screenshot';
import screenshots from '@site/src/data/screenshots.json';
import styles from './styles.module.css';

/**
 * Data-driven showcase — the route/state list lives in
 * `src/data/screenshots.json`, the same list
 * `scripts/capture_docs_screenshots.py` reads to know what to capture.
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
      </div>
    </section>
  );
}
