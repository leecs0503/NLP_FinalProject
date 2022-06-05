import Main from './main';
import { BrowserRouter, Route } from 'react-router-dom';

function App() {
  return (
    <Main />
    // <BrowserRouter>
    //   <Route exact path="/" component={Main} />
    //   {/* <Route path="/menu" component={Menu} /> */}
    // </BrowserRouter>
  );
}

export default App;
