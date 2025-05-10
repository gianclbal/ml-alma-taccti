import './../App.css';
import "bootstrap/dist/css/bootstrap.min.css";
import "./../css/dashboard.css";

import LoginScreen from './LoginScreen';
import DashboardWrapper from './DashboardWrapper'
//import Profile from './Profile';


function App() {
  return (
    <div className="App">
      <LoginScreen/>
      <DashboardWrapper/>
      
    </div>
  );
}

export default App;
