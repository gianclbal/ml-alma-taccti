import React from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import navbarlogo from './../images/navbarlogo.png'
// component import
import Tacit from './Tacit';

const DashboardWrapper = () => {
  const { logout, isAuthenticated, user } = useAuth0();

  return (
    isAuthenticated && (
      <div>
        <nav className="navbar navbar-expand-sm mt-1 mb-2">
          <div className="container-fluid">
            <a href="/">
              <span className="navbar-span navbar-brand">
                <img src={navbarlogo} className="d-inline-block align-top" alt="" loading="lazy"></img>
              </span>
            </a>
            <button
              className="navbar-toggler"
              type="button"
              data-toggle="collapse"
              data-target="#navbarNav"
            >
              <span className="navbar-toggler-icon" />
            </button>
            <div className="collapse navbar-collapse" id="navbarNav">
              <ul className="navbar-nav ml-auto">
                <li className="nav-item">
                  <span className="username">{user.name}</span>
                  <img className="d-inline-block align-top navbar-image" width="40"  src={user.picture} alt={user.name} />
                  <button
                    className="btn btn-primary btn-fill"
                    onClick={() => logout()}>
                    Logout
                  </button>
                </li>
              </ul>
            </div>
          </div>
        </nav>
        <Tacit />
      </div>
    )
  )
}


export default DashboardWrapper