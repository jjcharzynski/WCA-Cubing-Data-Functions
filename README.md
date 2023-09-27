# WCA Database Downloader and Analyzer

## Allowed Use

The code uses World Cubing Association competition results data. The WCA allows this information to be re-published, in whole or in part, as long as users are clearly notified of the following:

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

This function will download the data, extract it, and create a local SQLite database with the information.

### Analyzing Personal Progress
You can analyze the personal progress of a specific person and event using the calculatepersonalprogressbyevent() function and visualize it with the plotpersonalprogressbyevent() function.

```python
calculatepersonalprogressbyevent(personId, eventId)
plotpersonalprogressbyevent(personId, eventId)
```

### Comparing Cubers
To compare the progress of multiple cubers for a specific event, use the comparecubersbyevent() function.

```python
comparecubersbyevent(eventId, personId1, personId2, ...)
```

Cuber vs. Everybody
Compare a specific cuber's progress with others for a given event using the cubervseverybody() function.

```python
cubervseverybody(eventId, personId, personId1, personId2, ...)
```

## Contributing
If you would like to contribute to this project, feel free to open issues or submit pull requests on the GitHub repository.

## License
This code is licensed under the MIT License. See the LICENSE file for details.
