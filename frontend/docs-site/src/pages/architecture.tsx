import Layout from '@theme/Layout';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import useBaseUrl from '@docusaurus/useBaseUrl';
import React, {type JSX} from 'react';

import diagramGroups from '@site/src/data/architecture-diagrams.json';
import styles from './architecture.module.css';

/**
 * Interactive architecture diagrams.
 *
 * Each diagram is a hand-authored Archify spec under
 * docs-site/architecture-diagrams/specs/, built from real repo evidence
 * (src/routes/, src/lib/api.ts, the controllers, the slot registry,
 * nginx.conf, docker-compose.yml) and regenerated into
 * docs-site/static/architecture/*.html by
 * scripts/generate-architecture-diagrams.sh. Treat a diagram going stale
 * the same way as any other doc: fix the spec when the code it describes
 * changes. Group/diagram metadata lives in
 * src/data/architecture-diagrams.json, not here, so a cloned docs site
 * only needs to edit data.
 */

type Diagram = {
  id: string;
  title: string;
  description: string;
  height: number;
};

type Group = {
  id: string;
  label: string;
  diagrams: Diagram[];
};

const GROUPS = diagramGroups as Group[];
const DEFAULT_FRAME_HEIGHT = 1000;

function DiagramFrame({id, title, height}: {id: string; title: string; height: number}): JSX.Element {
  return (
    <div className={styles.frame}>
      <iframe
        src={useBaseUrl(`/architecture/${id}.html`)}
        title={title}
        className={styles.iframe}
        style={{height: `${height ?? DEFAULT_FRAME_HEIGHT}px`}}
      />
    </div>
  );
}

function GroupPanel({group}: {group: Group}): JSX.Element {
  return (
    <Tabs groupId={`architecture-${group.id}`} className={styles.innerTabs}>
      {group.diagrams.map((d) => (
        <TabItem key={d.id} value={d.id} label={d.title}>
          <p className={styles.diagramDescription}>{d.description}</p>
          <DiagramFrame id={d.id} title={d.title} height={d.height} />
        </TabItem>
      ))}
    </Tabs>
  );
}

export default function Architecture(): JSX.Element {
  return (
    <Layout
      title="Architecture"
      description="Interactive, evidence-based architecture diagrams for Cropwright: system overview, frontend modules, the labeling loop, ingest upload, and the review assign/undo sequence."
    >
      <header className={styles.hero}>
        <h1 className={styles.title}>Architecture</h1>
        <p className={styles.subtitle}>
          Interactive diagrams built from the real codebase. Pan, zoom, search, and click
          through guided views inside each one. Every diagram is a validated spec checked
          against the actual routes, controllers, and wire contract — not a hand-drawn
          illustration.
        </p>
      </header>

      <main className={styles.main}>
        <Tabs groupId="architecture-section" className={styles.outerTabs}>
          {GROUPS.map((group) => (
            <TabItem key={group.id} value={group.id} label={group.label}>
              <GroupPanel group={group} />
            </TabItem>
          ))}
        </Tabs>
      </main>

      <footer className={styles.footnote}>
        <p>
          Every diagram here is rendered by{' '}
          <a href="https://github.com/tt-a1i/archify">Archify</a> (MIT, © 2026 tt-a1i / Archify,
          © 2025 Cocoon AI).
        </p>
      </footer>
    </Layout>
  );
}
