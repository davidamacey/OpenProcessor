import React from 'react';
import Link from '@docusaurus/Link';
import workflow from '@site/src/data/workflow.json';
import {siteConfig} from '@site/site.config';
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
          {siteConfig.howItWorksMore.prefix}{' '}
          <Link to={siteConfig.howItWorksMore.to}>{siteConfig.howItWorksMore.label}</Link>.
        </p>
      </div>
    </section>
  );
}
