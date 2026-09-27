import React from 'react';
import clsx from 'clsx';
import roadmapData from '@site/src/data/roadmap.json';
import styles from './styles.module.css';

type Release = {
  version: string;
  headline: string;
  stage: string;
  summary: string;
  items: string[];
};

const STAGE_LABEL: Record<string, string> = {
  shipped: 'Shipped',
  now: 'In progress',
  next: 'Planned',
  later: 'Later',
};

/**
 * Data-driven roadmap. Cropwright has no issue-tracker generator (unlike
 * a sister project's roadmap.json, which a similar generator regenerates),
 * so `src/data/roadmap.json` is hand-maintained — edit the JSON, never this
 * component, when scope changes.
 */
export default function RoadmapView(): React.JSX.Element {
  const releases = roadmapData.releases as Release[];
  return (
    <div className={styles.timeline}>
      <p className={styles.note}>{roadmapData.note}</p>
      {releases.map((r) => (
        <div key={r.version} className={styles.release}>
          <div className={styles.releaseHeader}>
            <span className={clsx(styles.stage, styles[`stage-${r.stage}`])}>
              {STAGE_LABEL[r.stage] ?? r.stage}
            </span>
            <h2>{r.version}</h2>
          </div>
          <h3>{r.headline}</h3>
          <p>{r.summary}</p>
          <ul>
            {r.items.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
