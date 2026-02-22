# LiarMP4 Landing Page

This folder contains the static landing page for the **liarMP4** project. It is designed to be deployed entirely independently of the main React/Go application backend via GitHub Pages.

## Deployment Instructions (GitHub Pages)

Because this landing page exists in its own directory, you have a few options for deploying it.

### Option 1: Separate Repository (Recommended)
If you want to host this on `username.github.io/liarmp4-landing`:

1. Create a new repository on GitHub (e.g., `liarMP4-website`).
2. Copy the contents of this `landing/` folder into the root of that new repository.
3. Push to the `main` branch.
4. Go to the repository **Settings > Pages**.
5. Under "Source", select `Deploy from a branch`, choose `main`, and select the `/ (root)` folder.
6. Click **Save**. Your site will be live in a few minutes.

### Option 2: Deploy from a subfolder using GitHub Actions
If you are pushing this directly into your existing main codebase and want to deploy *only* the `landing` folder to GitHub Pages:

1. Create a workflow file in your repo at `.github/workflows/deploy-landing.yml`.
2. Add the following YAML configuration to build and deploy just the `landing` directory:

```yaml
name: Deploy Landing Page to GitHub Pages

on:
  push:
    branches: ["main"]
    paths:
      - 'landing/**'
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Setup Pages
        uses: actions/configure-pages@v4
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: './landing'
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

3. Go to repository **Settings > Pages** and ensure the Source is set to **GitHub Actions**.

## Technologies Used
- HTML5
- Tailwind CSS (via CDN)
- FontAwesome (Icons)
- Google Fonts (Inter, Fira Code)