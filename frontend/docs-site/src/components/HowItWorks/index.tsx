import React from 'react';
import Link from '@docusaurus/Link';
import workflow from '@site/src/data/workflow.json';
import styles from './styles.module.css';

export default function HowItWorks(): React.JSX.Element {
  return (
    <section className={styles.section}>
      <div className="container">
        <h2 className={styles.heading}>How it works</h2>
        <div className={styles.steps}>
          {workflow.map((w, i) => (
            <div key={w.step} className={styles.step}>
              <div className={styles.index}>{i + 1}</div>
              <div>
                <h4>
                  {w.step} <code>{w.route}</code>
                </h4>
                <p>{w.desc}</p>
              </div>
            </div>
          ))}
        </div>
        <p className={styles.more}>
          See the full <Link to="/architecture">workflow diagram</Link>.
        </p>
      </div>
    </section>
  );
}
