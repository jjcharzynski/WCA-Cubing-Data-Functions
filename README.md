# WCA Database Downloader and Analyzer

## Allowed Use

The code uses WCA cubing data. The WCA allows this information to be re-published, in whole or in part, as long as users are clearly notified of the following:

> This information is based on competition results owned and maintained by the
> World Cube Association, published at [https://worldcubeassociation.org/results](https://worldcubeassociation.org/results).

## Table of Contents
- [Description](#description)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Downloading WCA Data](#downloading-wca-data)
  - [Analyzing Personal Progress](#analyzing-personal-progress)
  - [Comparing Cubers](#comparing-cubers)
  - [Cuber vs. Everybody](#cuber-vs-everybody)
- [Contributing](#contributing)
- [License](#license)

## Description

This Python script allows you to download and analyze competition data from the World Cube Association (WCA) database. It can create a local SQLite database with WCA data, retrieve and process competition data for specific persons and events, calculate and plot personal progress, and compare progress between cubers.

## Requirements

To run this code, you need:

- Python 3.x
- Required Python packages (install using `pip install package_name`):
  - requests
  - zipfile
  - json
  - pandas
  - sqlalchemy
  - matplotlib
  - numpy
  - scikit-learn (sklearn)
  - scipy

## Installation

1. Clone or download this repository to your local machine.

2. Install the required Python packages as mentioned in the Requirements section.

## Usage

### Downloading WCA Data

Before analyzing data, you need to download the latest WCA competition data. You can do this using the `createWCAdatabase()` function in the script.

```python
createWCAdatabase()
```

bob
