# WebPack Learning Notes

## Installation and setup

Check your node.js, npm and yarn version

``` shell
node -v
npm -v
yarn -v
```

Update npm if necessary

``` shell
npm i -g npm
```

### Initialize workspace using yarn

``` shell
yarn init
```

Install WebPack

``` shell
yarn add webpack --dev
yarn add @type/webpack
```

### Build project

``` shell
webpack ./src/app.js ./dist/app.bundle.js

# build product version with boiler plates removed and minified
webpack ./src/app.js ./dist/app.bundle.js -p

# watch mode: re-bundle if any change
webpack ./src/app.js ./dist/app.bundle.js --watch
```

## WebPack Configuration

Example webpack.config.js

``` javascript
module.exports = {
  entry: [
    './main.js',
  ],
  output: {
    path: __dirname,
    filename: 'main.bundle.js',
  },
  devtool: "#eval-source-map",
  module: {
    loaders: [
      {
        test: /\.jsx?$/,
        exclude: /node_modules/,
        loaders: ['babel-loader'],
      },
    ],
  },
};
```

Example index.html

``` html
<html>
  <head>
    <meta charset='UTF-8' />
  </head>
  <body>
    <h1>Open your console.</h1>
    <script type="module">
      import { main } from './main.js';
      main();
    </script>
    <script nomodule type="text/javascript"src="bundle.js"></script>
  </body>
</html>
```

> Notice that the `type="module"` allows using ES6 `import` in `<script>` tag.
> And if the browser doesn't support `module`, it will fallback to the `nomodule`script.

## Setup Babel Loader

1. Install Babel modules

``` shell
npm install --save-dev babel-cli babel-preset-es2015 babel-loader
```

2. Create a Babel configuration file named `.babelrc`:

``` json
// .babelrc
{
"presets": ["es2015"]
}
```

3. Configure `webpack.config.js` to use Babel loader:

``` javascript
const path = require('path');

module.exports = {
  entry: ['babel-polyfill', './index.js'],
  output: {
    filename: 'bundle.js',
    path: path.resolve(__dirname)
  },  module: {
    rules: [
      {
        test: /.js$/,
        exclude: /node_modules/,
        use: 'babel-loader'
      }
    ]
  }
};
```

4. Add a webpack build command in `package.json` file:
``` json
{
  /* package.json configuration */

  "scripts": {
    "build": "webpack --config webpack.config.js",
  }

  /* remaining properties */
}
```

Then run `npm run build` will generate a transpiled and bundled js file for browers that don't supports ES2015.

yarn global add webpack
yarn global add webpack-cli
yarn global add webpack-dev-server

yarn add react@16.0.1 react-dom@16.0.1 webpack-dev-server@3.1.14 live-server@1.2.1 extract-text-webpack-plugin@next
yarn add -D webpack-cli

npm install webpack webpack-cli webpack-dev-server --save-dev

