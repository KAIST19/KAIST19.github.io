import React from 'react';
import logo from './img/logo.png';
import headingImg from './img/meet_our_designer.png';
import searchIcon from './img/search_icon.png';
import './App.css';

const App = () => (
  <div className="App">
    <div className="flex-container">
      <Navbar logo={logo} />
      <Header headingImg={headingImg} searchIcon={searchIcon} />
    </div>
  </div>
);

const Navbar = ({ logo }) => (
  <nav className="navbar">
    <img src={logo} alt="Logo" className="navbar-logo" />
    <div>
      <button className="btn btn-outline-light">Login</button>
      <button className="btn btn-outline-light">Sign up</button>
    </div>
  </nav>
);

const Header = ({ headingImg, searchIcon }) => (
  <header className="header">
    <img src={headingImg} alt="Meet our designer" className="heading-image" />
    <SearchBar searchIcon={searchIcon} />
  </header>
);

const SearchBar = ({ searchIcon }) => (
  <div className="search-bar">
    <form className="d-flex">
      <input className="search-input" type="search" placeholder="What are you looking for?" aria-label="Search" />
      <button className="search-button" type="submit">
        <img src={searchIcon} alt="Search icon" />
      </button>
    </form>
  </div>
);

export default App;
