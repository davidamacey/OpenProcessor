import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import {siteConfig} from '@site/site.config';
import styles from './styles.module.css';

export default function Hero(): React.JSX.Element {
  const {siteConfig: docusaurusConfig} = useDocusaurusContext();
  return (
    <header className={styles.hero}>
      <div className="container">
        <div className={styles.badges}>
          <span className={styles.badge}>{siteConfig.license}</span>
          <span className={styles.badge}>Self-hosted</span>
          <span className={styles.badge}>Domain-agnostic</span>
        </div>
        <h1 className={styles.title}>{docusaurusConfig.title}</h1>
        <p className={styles.tagline}>{docusaurusConfig.tagline}</p>
        <div className={styles.actions}>
          <Link className="button button--primary button--lg" to="/docs/getting-started/introduction">
            Get started
          </Link>
          <Link className="button button--secondary button--lg" to={siteConfig.githubRepo}>
            View on GitHub
          </Link>
        </div>
      </div>
    </header>
  );
}
