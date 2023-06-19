# CONTRIBUTING.md

Thank you for your interest in contributing to `whynot`!

## Setup

To contribute to `whynot`, you'll need to set up a local development environment. Here's how:

1. **Fork the repository:** Fork the `whynot` repository to your GitHub account.
2. **Clone the repository:** Clone the forked repository to your local machine.

```bash
git clone https://github.com/{your_github_username}/whynot.git

```

3. **Create a new branch:** Create a new branch for your feature.

```bash
git checkout -b feature-branch-name

```

4. **Install the dependencies:** Currently we support `pypi` based packages. You can install these with the following command:

```bash
pip install -r requirements.txt

```

5. **Make changes:** Make your proposed changes to the codebase.

6. **Run tests:** Ensure that all tests pass.

```bash
pytest

```

7. **Push changes:** If everything looks good, push your changes to your fork.

```bash
git add .
git commit -m "Brief description of changes"
git push origin feature-branch-name

```

8. **Open a pull request:** Go back to your forked repository on GitHub, and open a new pull request to the main `whynot` repository.

## Code Standards

Please ensure your code adheres to the Python PEP8 style guide. This ensures consistency and readability across the project.

## Tests

Before submitting your changes, make sure all existing tests pass. If you've added a new feature, please include tests that cover your feature.

## Pull Requests

When you open a pull request, please be as clear and descriptive as possible in the title and the description.

## Reporting Issues

If you're facing any problems or have a feature request, please open an issue about it on GitHub. We appreciate your feedback and we'll respond as soon as we can.

## Questions?

If you have any queries or need further clarification, please feel free to reach out to us.
