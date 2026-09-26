import React from 'react';
import Layout from '@theme/Layout';
import DiagramSection from '@site/src/components/DiagramSection';
import {siteConfig} from '@site/site.config';

export default function Architecture(): React.JSX.Element {
  return (
    <Layout
      title="Architecture"
      description={siteConfig.architecturePage.description}>
      <main className="container margin-vert--lg">
        <h1>Architecture</h1>
        <p>{siteConfig.architecturePage.intro}</p>
        <DiagramSection />
      </main>
    </Layout>
  );
}
