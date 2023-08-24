import React, { useState } from 'react';
import logo from './img/logo.png';
import headingImg from './img/meet_our_designer.png';
import searchIcon from './img/search_icon.png';
import './App.css';

const App = () => {
  const [searchInput, setSearchInput] = useState("");
  const [searchResults, setSearchResults] = useState([]);

  const handleSearch = (e) => {
    e.preventDefault();
    // Mocked search results
    const results = ['Result 1', 'Result 2', 'Result 3'];
    setSearchResults(results);
  };

  return (
    <div className="App">
      <div className="flex-container">
        <Navbar logo={logo} />
        <Header headingImg={headingImg} />
        <SearchBar 
          searchIcon={searchIcon} 
          searchInput={searchInput} 
          onSearchInputChange={setSearchInput}
          onSearchSubmit={handleSearch} 
        />
        <div className="search-results">
          {searchResults.map((result, index) => <div key={index}>{result}</div>)}
        </div>
      </div>
    </div>
  );
};

const Navbar = ({ logo }) => (
  <nav className="navbar">
    <img src={logo} alt="Logo" className="navbar-logo" />
    <div>
      <button className="btn btn-outline-light">Login</button>
      <button className="btn btn-outline-light">Sign up</button>
    </div>
  </nav>
);

const Header = ({ headingImg }) => (
  <header className="header">
    <img src={headingImg} alt="Meet our designer" className="heading-image" />
  </header>
);

const SearchBar = ({ searchIcon, searchInput, onSearchInputChange, onSearchSubmit }) => (
  <div className="search-bar">
    <form className="d-flex" onSubmit={onSearchSubmit}>
      <input 
        className="search-input" 
        type="search" 
        placeholder="What are you looking for?" 
        aria-label="Search" 
        value={searchInput}
        onChange={e => onSearchInputChange(e.target.value)}
      />
      <button className="search-button" type="submit">
        <img src={searchIcon} alt="Search icon" />
      </button>
    </form>
  </div>
);

export default App;
