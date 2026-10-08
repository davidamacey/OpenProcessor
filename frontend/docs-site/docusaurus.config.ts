import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

import {siteConfig} from './site.config';

// This runs in Node.js - don't use client-side code here.
//
// Every project-specific value (title, repo, URLs, nav/footer, sibling
// links) is read from `site.config.ts`, not hardcoded here — this file
// should be nearly identical between Cropwright and a cloned sibling site.
// See docs-site/TEMPLATE.md.

const config: Config = {
  title: siteConfig.title,
  tagline: siteConfig.tagline,
  favicon: siteConfig.favicon,

  future: {
    v4: true,
  },

  url: siteConfig.url,
  baseUrl: siteConfig.baseUrl,

  organizationName: siteConfig.organizationName,
  projectName: siteConfig.projectName,

  // Fail the build on any broken internal link or heading anchor rather
  // than shipping a dead link. Keep this `throw` in any cloned sibling site.
  onBrokenLinks: 'throw',
  onBrokenAnchors: 'throw',

  customFields: {
    githubRepo: siteConfig.githubRepo,
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: siteConfig.editUrlBase,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: siteConfig.favicon,
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: false,
      disableSwitch: false,
    },
    announcementBar: {
      id: siteConfig.announcementBar.id,
      content: siteConfig.announcementBar.content,
      backgroundColor: siteConfig.announcementBar.backgroundColor,
      textColor: siteConfig.announcementBar.textColor,
      isCloseable: true,
    },
    navbar: {
      title: siteConfig.navbar.title,
      logo: {
        alt: `${siteConfig.title} Logo`,
        src: siteConfig.logo,
      },
      items: siteConfig.navbar.items,
    },
    footer: {
      style: 'dark',
      links: siteConfig.footerLinkGroups,
      copyright: `Copyright © ${new Date().getFullYear()} ${siteConfig.copyrightHolder}. ${siteConfig.title} is open source under the ${siteConfig.license} License.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
