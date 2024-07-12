# TACIT Front End

### Information
* This repository contains the front-end code for the TACIT app and has been bootstrapped using create-react-app.
* Please ensure that the node.js and npm are installed on your machine. If not then download it from [here](https://nodejs.org/en/download/)
* Copy the contents of the directory <strong style="color:red;">tacit-frontend</strong> to your machine.
* To install the dependencies and start the front end service on your machine run.
```bash
# change directory to tacit-frontend
cd tacit-frontend
# install the dependencies
npm install
# start the project
npm start
```
* The service will be available on http://localhost:3000/

### TACIT Home Page
![Welcome Screen](./../snapshots/welcome_screen.png)

### Authentication
* Authentication is handled using [Auth0](https://auth0.com/) which is an authentication service and can handle 7000 active users in the free plan.
* Sign up for a free plan and need to create an application.

![Create Application](./../snapshots/auth0_create_app.png)
* In the index.js need to update the <strong style="color:red;">domain</strong> and <strong style="color:red;">clientId</strong> values. Do not hard code them, use <strong style="color:red;">environment variables</strong> instead. 

![App Secret](./../snapshots/app_secret.png)

### API Call
* For the front-end to work, it needs an API endpoint which processes the uploaded essays. The endpoint should also be added as an <strong style="color:red;">environment variable</strong> and not hardcoded in the <strong style="color:red;">Tacit.js</strong> file.
```bash
// Need to specify the endpoint here.
// http://ec2-13-57-214-128.us-west-1.compute.amazonaws.com:5000
// Localhost url
// http://localhost:5000/culturalcapitals
fetch(process.env.API_ENDPOINT/culturalcapitals, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(data)
}).then((response) => {
   response.json().then((body) => {
```
### Dashboard
* Post Authentication
![Post Auth](./../snapshots/post_auth.png)

* Post Data Upload
![Post Upload](./../snapshots/post_upload_1.png)
![Post Upload](./../snapshots/post_upload_2.png)

### Code
* Tacit.js: Contains the code for the dashboard.
* LoginScreen.js: Contains the code for homepage.
* <i style="color:orange;">Code Improvements:</i> Improve aesthetics, functionality, additional testing