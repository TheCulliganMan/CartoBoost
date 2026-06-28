import fs from 'node:fs';
import path from 'node:path';
import type { Config, Plugin } from '@docusaurus/types';
import type { Options as ClassicOptions } from '@docusaurus/preset-classic';
import { themes as prismThemes } from 'prism-react-renderer';

function llmsTxtDocsPlugin(): Plugin {
  return {
    name: 'llms-txt-docs',
    postBuild({ outDir }) {
      const source = path.join(process.cwd(), 'docs', 'llms.txt');
      const target = path.join(outDir, 'docs', 'llms.txt');
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.copyFileSync(source, target);
    },
  };
}

type DocsVersion = {
  version: string;
  ref: string;
};

const docsVersionsPath = path.join(process.cwd(), 'docs-versions.json');
const docsVersions: DocsVersion[] = fs.existsSync(docsVersionsPath)
  ? JSON.parse(fs.readFileSync(docsVersionsPath, 'utf8'))
  : [];
const docsVersionItems = [
  { label: 'latest', to: '/docs' },
  ...docsVersions.map(({ version }) => ({
    label: version,
    to: `/docs/${version}`,
  })),
];

const config: Config = {
  title: 'CartoBoost',
  tagline: 'Temporal, spatial, geotemporal, and graph-aware regression',
  favicon: 'img/cartoboost-route-splitter-logo.svg',

  url: 'https://theculliganman.github.io',
  baseUrl: '/CartoBoost/',
  organizationName: 'TheCulliganMan',
  projectName: 'CartoBoost',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onDuplicateRoutes: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },

  presets: [
    [
      'classic',
      {
        docs: {
          path: 'docs',
          routeBasePath: 'docs',
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/TheCulliganMan/CartoBoost/edit/main/',
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
          exclude: ['README.md', 'llms.txt', '**/AGENTS.md', 'assets/**'],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
        sitemap: {
          changefreq: 'weekly',
          priority: 0.7,
          ignorePatterns: ['/tags/**'],
          filename: 'sitemap.xml',
        },
      } satisfies ClassicOptions,
    ],
  ],

  themes: [
    '@docusaurus/theme-mermaid',
    [
      '@easyops-cn/docusaurus-search-local',
      {
        docsRouteBasePath: 'docs',
        hashed: 'filename',
        indexBlog: false,
        indexDocs: true,
        indexPages: true,
        language: ['en'],
        searchBarPosition: 'right',
      },
    ],
  ],

  plugins: [
    llmsTxtDocsPlugin,
    [
      '@docusaurus/plugin-client-redirects',
      {
        redirects: [
          { from: '/installation', to: '/docs/installation' },
          { from: '/getting-started', to: '/docs/getting-started' },
          { from: '/feature_catalog', to: '/docs/feature_catalog' },
          { from: '/forecasting', to: '/docs/forecasting' },
          { from: '/docs/forecasting_api', to: '/docs/forecasting' },
          { from: '/docs/forecasting_artifacts', to: '/docs/forecasting' },
          { from: '/docs/forecasting_backtesting', to: '/docs/forecasting' },
          { from: '/docs/forecasting_cli', to: '/docs/forecasting' },
          { from: '/docs/forecasting_decomposition', to: '/docs/user-guide/forecasting-models/piecewise-linear-seasonal' },
          { from: '/docs/forecasting_direct', to: '/docs/user-guide/forecasting-models/auto-forecaster' },
          { from: '/docs/forecasting_ensemble', to: '/docs/user-guide/forecasting-models/ensembles' },
          { from: '/docs/forecasting_examples', to: '/docs/user-guide/forecasting-models' },
          { from: '/docs/forecasting_frame', to: '/docs/forecasting' },
          { from: '/docs/forecasting_hybridization_assessment', to: '/docs/user-guide/forecasting-models/auto-forecaster' },
          { from: '/docs/forecasting_lag_features', to: '/docs/user-guide/forecasting-models/cartoboost-lag' },
          { from: '/docs/forecasting_models', to: '/docs/user-guide/forecasting-models' },
          { from: '/docs/user-guide/forecasting-models/arima-examples', to: '/docs/user-guide/forecasting-models/arima' },
          { from: '/docs/user-guide/forecasting-models/prophet', to: '/docs/plotting' },
          { from: '/docs/forecasting_neural', to: '/docs/forecasting' },
          { from: '/docs/forecasting_overhaul', to: '/docs/user-guide/forecasting-models/auto-forecaster' },
          { from: '/docs/forecasting_probabilistic', to: '/docs/forecasting' },
          { from: '/docs/forecasting_reconciliation', to: '/docs/forecasting' },
          { from: '/benchmarks', to: '/docs/benchmarks' },
          { from: '/reference/python-api', to: '/docs/reference/python-api' },
          { from: '/reference/cli', to: '/docs/reference/cli' },
          { from: '/user-guide/model-types', to: '/docs/user-guide/model-types' },
          { from: '/user-guide/python-estimator', to: '/docs/user-guide/python-estimator' },
          { from: '/forecast-lab', to: '/modeling-lab' },
        ],
      },
    ],
  ],

  themeConfig: {
    image: 'img/social-card.svg',
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'CartoBoost',
      logo: {
        alt: 'CartoBoost',
        src: 'img/cartoboost-route-splitter-logo.svg',
      },
      items: [
        { to: '/docs/installation', label: 'Get Started', position: 'left' },
        { to: '/modeling-lab', label: 'Modeling Lab', position: 'left' },
        { to: '/docs/reference/python-api', label: 'Reference', position: 'left' },
        { to: '/docs/benchmarks', label: 'Benchmarks', position: 'left' },
        {
          type: 'dropdown',
          label: 'Docs',
          position: 'right',
          items: docsVersionItems,
        },
        { type: 'search', position: 'right' },
        {
          href: 'https://github.com/TheCulliganMan/CartoBoost',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Start',
          items: [
            { label: 'Install', to: '/docs/installation' },
            { label: 'Getting Started', to: '/docs/getting-started' },
            { label: 'Choose A Model', to: '/docs/user-guide/model-types' },
          ],
        },
        {
          title: 'Deep Dives',
          items: [
            { label: 'Forecasting', to: '/docs/forecasting' },
            { label: 'Boosting Models', to: '/docs/user-guide/boosting-models' },
            { label: 'Graph Models', to: '/docs/user-guide/graph-models' },
            { label: 'Neural Models', to: '/docs/user-guide/neural-models' },
          ],
        },
        {
          title: 'Evidence',
          items: [
            { label: 'Benchmarks', to: '/docs/benchmarks' },
            { label: 'Feature Catalog', to: '/docs/feature_catalog' },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} CartoBoost contributors.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'toml', 'rust'],
    },
  },
};

export default config;
