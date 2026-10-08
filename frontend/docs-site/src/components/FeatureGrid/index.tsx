import React from 'react';
import features from '@site/src/data/features.json';
import styles from './styles.module.css';

/**
 * Renders `src/data/features.json` — no feature copy lives in this
 * component. A cloned sibling site only needs to replace that JSON.
 */
export default function FeatureGrid(): React.JSX.Element {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.grid}>
          {features.map((f) => (
            <div key={f.title} className={styles.card}>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
