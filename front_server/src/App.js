import Main from './main';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <Route exact path="/" component={Main} />
      {/* <Route path="/menu" component={Menu} /> */}
    </BrowserRouter>
  );
}

export default App;
