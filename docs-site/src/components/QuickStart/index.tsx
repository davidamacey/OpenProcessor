import React from 'react';
import CodeBlock from '@theme/CodeBlock';
import Link from '@docusaurus/Link';
import {siteConfig} from '@site/site.config';
import styles from './styles.module.css';

export default function QuickStart(): React.JSX.Element {
  return (
    <section className={styles.section}>
      <div className="container">
        <h2>Quick start</h2>
        <p>{siteConfig.quickStart.intro}</p>
        <CodeBlock language="bash">{siteConfig.quickStart.command}</CodeBlock>
        <p>
          <Link to={siteConfig.quickStart.guidePath}>{siteConfig.quickStart.guideLabel}</Link>
        </p>
      </div>
    </section>
  );
}
