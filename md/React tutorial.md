# React Tutorial

Link
<https://reactjs.org/tutorial/tutorial.html>

## Setup

npx create-react-app tic-tac-toe

## Component

``` jsx
class ShoppingList extends React.Component {
  render() {
    return (
      <div className="shopping-list">
        <h1>Shopping List for {this.props.name}</h1>
        <ul>
          <li>Instagram</li>
          <li>WhatsApp</li>
          <li>Oculus</li>
        </ul>
      </div>
    );

    // equivalent to
    return React.createElement('div', {className: 'shopping-list'},
      React.createElement('h1', /* ... h1 children ... */),
      React.createElement('ul', /* ... ul children ... */)
    );
  }
}
```

## Lift state up

To collect data from multiple children, or to have two child components communicate with each other, you need to declare the **shared state** in their parent component instead. The parent component can pass the **state** back down to the children by using **props**; this keeps the child components in sync with each other and with the parent component.

## ES6 syntax candy: Object spreading

``` javascript
// traditional javascript using Object.assign
var newPlayer = Object.assign({}, player, {score: 2});
// es6 object spreading
var newPlayer = {...player, score: 2};

props = {a: 'a', b: 'b'}
context = {a: 'a1', c:'c'}

p = {...props, context}
{a: "a", b: "b", context: {…}}

p = {...context, ...props}
{a: "a", c: "c", b: "b"}

p = {...props, ...context}
{a: "a1", b: "b", c: "c"}
```

React control examples
http://jsfiddle.net/infiniteluke/908earbh/9/
https://reactabular.js.org/#/
https://github.com/azizali/react-super-treeview
http://react-redux-grid.herokuapp.com/Tree
https://reactjsexample.com/a-react-grid-tree-component-written-in-the-redux-pattern/
https://github.com/bencripps/react-redux-grid

