# Building Applications with React and Redux in ES6

## Link

<https://app.pluralsight.com/library/courses/react-redux-react-router-es6/table-of-contents>

## Content covered

* Redux
* ES6 with Babel
* React Router
* Webpack
* npm scripts
* ESLint
* Mocha, React Test Urils, and Enzyme
* Webstorm
* Plauralsight Admin App

## Boiler plate

<https://github.com/coryhouse/react-slingshot>
<https://github.com/coryhouse/pluralsight-redux-starter>

### Hot Reloading

babel-preset-react-hmre

## React component

``` javascript
// ES5 Create Class Component
var HelloWorld = React.createClass({
  render: function () {
    return (
      <h1>Hello World</h1>
    );
  }
});

// ES6 Class Component
class HelloWorld extends React.Component {
  constructor(props) {
    super(props);
  }
  render() {
    return (
      <h1>Hello World</h1>
    );
  }
}
```

### React in ES6: Stateless Functional Components

* No autobind
* PropTypes declared separately
* Default props declared separately
* Set initial state in constructor

``` javascript
// ES5 Stateless Functional Component
var HelloWorld = function(props) {
  return (
    <h1>Hello World</h1>
  );
});

// ES6 Stateless Functional Component
const HelloWorld = (props) => {
  return (
    <h1>Hello World</h1>
  );
});
```

### Best Practice

A suggested src structure is like below:

``` console
\node_modules\
\src\
  actions\
  api\
  components\
    about\
    common\
    course\
    home\
    App.js
  reducers\
    courseReducer.js
    index.js
  store\
    configurationStore.js
  styles\
  index.html
  index.js
  index.test.js
  routes.js
\tools\
.babelrc
.editorconfig
.eslintrc
package.json
webpack.config.js
webpack.config.dev.js
```

### Redux Async Libraries

* redux-thunk
* redux-promise
* redux-sage
