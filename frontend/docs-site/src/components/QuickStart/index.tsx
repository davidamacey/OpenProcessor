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
        <p>Point Cropwright at a running OpenProcessor backend and you're labeling.</p>
        <CodeBlock language="bash">
          {`git clone ${siteConfig.githubRepo} && cd cropwright\ncp .env.example .env\n# edit .env: API_UPSTREAM, PUBLIC_API_PREFIX, OP_DOCKER_NETWORK\ndocker compose up -d --build`}
        </CodeBlock>
        <p>
          <Link to="/docs/getting-started/quick-start">Full quick-start guide</Link>
        </p>
      </div>
    </section>
  );
}
