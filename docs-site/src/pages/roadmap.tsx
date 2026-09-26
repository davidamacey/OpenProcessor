import React from 'react';
import Layout from '@theme/Layout';
import RoadmapView from '@site/src/components/RoadmapView';
import {siteConfig} from '@site/site.config';

export default function Roadmap(): React.JSX.Element {
  return (
    <Layout title="Roadmap" description={siteConfig.roadmapPage.description}>
      <main className="container margin-vert--lg">
        <h1>Roadmap</h1>
        <RoadmapView />
      </main>
    </Layout>
  );
}
