## Allowed Use
The code uses WCA cubing data. The WCA allows this information to be re-published, in whole or in part, as long
as users are clearly notified of the following:

> This information is based on competition results owned and maintained by the
> World Cube Assocation, published at https://worldcubeassociation.org/results

## WCA Database Downloader and Analyzer
Table of Contents
Description
Requirements
Installation
Usage
Downloading WCA Data
Analyzing Personal Progress
Comparing Cubers
Cuber vs. Everybody
Contributing
License
Description
This Python script allows you to download and analyze competition data from the World Cube Association (WCA) database. It can create a local SQLite database with WCA data, retrieve and process competition data for specific persons and events, calculate and plot personal progress, and compare progress between cubers.

## Requirements
To run this code, you need:

Python 3.x
Required Python packages (install using pip install package_name):
requests
zipfile
json
pandas
sqlalchemy
matplotlib
numpy
scikit-learn (sklearn)
scipy
Installation
Clone or download this repository to your local machine.

Install the required Python packages as mentioned in the Requirements section.

Usage
Downloading WCA Data
Before analyzing data, you need to download the latest WCA competition data. You can do this using the createWCAdatabase() function in the script.

python
Copy code
createWCAdatabase()
This function will download the data, extract it, and create a local SQLite database with the information.

Analyzing Personal Progress
You can analyze the personal progress of a specific person and event using the calculatepersonalprogressbyevent() function and visualize it with the plotpersonalprogressbyevent() function.

python
Copy code
calculatepersonalprogressbyevent(personId, eventId)
plotpersonalprogressbyevent(personId, eventId)
Comparing Cubers
To compare the progress of multiple cubers for a specific event, use the comparecubersbyevent() function.

python
Copy code
comparecubersbyevent(eventId, personId1, personId2, ...)
Cuber vs. Everybody
Compare a specific cuber's progress with others for a given event using the cubervseverybody() function.

python
Copy code
cubervseverybody(eventId, personId, personId1, personId2, ...)
Contributing
If you would like to contribute to this project, feel free to open issues or submit pull requests on the GitHub repository.

License
This code is licensed under the MIT License. See the LICENSE file for details.

Feel free to customize the README file further to include any additional information or instructions specific to your use case.
