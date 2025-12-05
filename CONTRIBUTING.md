# Contributing Guidelines

Thank you for your interest in contributing to py4dgeo!
We welcome contributions of new methods and usage demonstrations that help users better understand and apply py4dgeo functionality.

This document outlines the recommended workflow for contributing code, examples, or documentation to the project.

🌟 **What You Can Contribute**

You are encouraged to contribute:

New methods that extend py4dgeo’s analytical capabilities

Usage examples and application demos that illustrate how to apply existing or new methods

If you want to:

1) Implement a new method, or

2) Provide a new example demonstrating py4dgeo functionality,

please follow the steps below.

📄 **Contribution Workflow**

1. **Fork the repository and create a branch** for your contribution. This keeps your work isolated and makes review easier.
2. **Set up a local development environment**: Before preparing demos or implementing new methods, please familiarize yourself with the py4dgeo development environment.
Consider the instructions given in the Installation section of the [README](README.md) to install py4dgeo in editable mode along with development tools.
Running pre-commit and pytest locally helps you avoid CI failures.
3. **Prepare your method/demo** using the templates provided in the [`contributor_template/`](contributor_template/) directory:
   - [`basic_usage.ipynb`](contributor_template/basic_usage.ipynb): Use this notebook to document your newly implemented method.
It will be integrated into the official documentation [readthedocs](https://py4dgeo.readthedocs.io/en/latest/basic.html).
   - [`application_demo.ipynb`](contributor_template/application_demo.ipynb): Use this notebook to demonstrate practical use cases or applications of a method.

4. **Add yourself to the contributor list**: Please add your name to the [COPYING.md](../COPYING.md) file under the contributors name list. This helps us keep track of project contributors and acknowledge your work in software publications.
   Example entry: 

   `- Your Name, YYYY-YYYY - contribution summary (e.g., "Implemented new method X", "Added application demo Y")`

5. **Open a pull request**: Once your branch is ready, push it to your fork and open a Pull Request (PR) to the main py4dgeo repository.
   Your PR should include:
   - a one-paragraph summary of what your contribution does
   - a description of changes, including new or modified code, added or updated tests, documentation updates
   - instructions for running your demo and tests
   - any open questions, limitations, or follow-up ideas

6. **Address review feedback** (if applicaple): A maintainer may request changes during the review process. Please address these through follow-up commits pushed to the same branch.
   When checks are green and reviews are satisfied, a maintainer will merge.

7. **Finalization and merge**: Your contribution will be merged once all automated checks (tests, formatting, docs build) pass and reviewers have approved your changes.
After merge, **verify** that your docs/examples appear correctly in the next documentation build (ReadTheDocs).


❓ For any issues or questions during any of the steps, do not hesitate to open an issue on the [issue tracker](https://github.com/3dgeo-heidelberg/py4dgeo/issues).
