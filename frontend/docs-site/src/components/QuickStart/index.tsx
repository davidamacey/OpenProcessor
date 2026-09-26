import React from 'react';
import CodeBlock from '@theme/CodeBlock';
import Link from '@docusaurus/Link';
import {siteConfig} from '@site/site.config';
import styles from './styles.module.css';

export default function QuickStart(): React.JSX.Element {
  const raw = `https://raw.githubusercontent.com/${siteConfig.organizationName}/${siteConfig.projectName}/main`;
  return (
    <section className={styles.section}>
      <div className="container">
        <h2>Quick start</h2>
        <p>
          Point Cropwright at a running OpenProcessor backend and you're labeling. No clone
          needed: the published image runs on amd64 and arm64.
        </p>
        <CodeBlock language="bash">
          {`curl -fsSLO ${raw}/docker-compose.yml\ncurl -fsSL ${raw}/.env.example -o .env\n# edit .env: API_UPSTREAM, PUBLIC_API_PREFIX, OP_DOCKER_NETWORK\ndocker compose pull && docker compose up -d`}
        </CodeBlock>
        <p>
          <Link to="/docs/getting-started/quick-start">Full quick-start guide</Link>
        </p>
      </div>
    </section>
  );
}
