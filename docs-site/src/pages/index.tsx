import React from 'react';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

import Hero from '@site/src/components/Hero';
import FeatureGrid from '@site/src/components/FeatureGrid';
import HowItWorks from '@site/src/components/HowItWorks';
import QuickStart from '@site/src/components/QuickStart';
import ScreenshotShowcase from '@site/src/components/ScreenshotShowcase';
import backendScreenshots from '@site/src/data/backend_screenshots.json';

export default function Home(): React.JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <Hero />
      <main>
        <FeatureGrid />
        <HowItWorks />
        <ScreenshotShowcase />
        <ScreenshotShowcase
          heading="See the backend in action"
          items={backendScreenshots}
          credits={
            <>
              Captured from a stack holding only public sample data (
              <a href="https://cocodataset.org">COCO</a> val2017). The tools shown are Swagger UI,
              Prometheus, MLflow and OpenSearch Dashboards, each under its own license.{' '}
              <a href="docs/developer-guide/screenshots#backend-screenshots">Credits</a>
            </>
          }
        />
        <QuickStart />
      </main>
    </Layout>
  );
}
