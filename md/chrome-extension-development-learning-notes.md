# Chrome Extension Development Learning Notes

https://developer.chrome.com/extensions/getstarted

## Manifest file

1. Create a manifest.json file like following:
``` json
{
  "name": "Getting Started Example",
  "version": "1.0",
  "description": "Build an Extension!",
  "permissions": ["activeTab", "declarativeContent", "storage"],
  "options_page": "options.html",
  "background": {
    "scripts": ["background.js"],
    "persistent": false
  },
  "page_action": {
    "default_popup": "popup.html",
    "default_icon": {
      "32": "images/get_started32.png"
    }
  },
  "icons": {
    "32": "images/get_started32.png"
  },
  "manifest_version": 2
}
```
2. In Chrome, go to `chrome://extensions`;
3. Enable `Developer Mode`, click `LOAD UNPACKED` button and select the extension directory;

The extension will be loaded and its source code could be debugged in developer tool.
Each file can be accessed using a URL like: `chrome-extension://<extensionID>/<pathToFile>`. The extension ID could be found on extensions manager page.

## Background Script

Background script is the extension's event handler. It contains listeners for browser event that are important to the extension. An example of background.js is like following. This script save shared value `color: '#3aa757'` via storage API, and show page action only when the content page is on `developer.chrome.com` site.

``` javascript
'use strict';
chrome.runtime.onInstalled.addListener(function() {
  chrome.storage.sync.set({color: '#3aa757'}, function() {
    console.log('The color is green.');
  });
  chrome.declarativeContent.onPageChanged.removeRules(undefined, function() {
    chrome.declarativeContent.onPageChanged.addRules([{
      conditions: [new chrome.declarativeContent.PageStateMatcher({
        pageUrl: {hostEquals: 'developer.chrome.com'},
      })],
      actions: [new chrome.declarativeContent.ShowPageAction()]
    }]);
  });
});
```

> Note: You should always register or unregister rules in bulk rather than individually because each of these operations recreates internal data structures. This re-creation is computationally expensive but facilitates a faster matching algorithm.

### chrome.runtime API
https://developer.chrome.com/apps/runtime

Retrieve the background page, return details about the manifest, and listen for and respond to events in the app or extension lifecycle.

Methods

* getBackgroundPage
* getManifest
* getURL
* connect
* sendMessage
* getPlatformInfo
* getPackageDirectoryEntry
* ...

Events

* onStartup
* onInstalled
* onConnect
* onMessage
* ...

### chrome.storage API
https://developer.chrome.com/apps/storage

### chrome.storage API
https://developer.chrome.com/extensions/declarativeContent

## UI Elements

Extensions's UI could be browser action or page action, or context menus, omnibox, or keyboard shortcut.

This example contains a page action using popup. It is registered in manifest like following:
``` json
  "page_action": {
    "default_popup": "popup.html",
```

Use any HTML+JavaScript in the popup.html file. It sets the content page background color to be what we set in the `background.js`.

### popup.html
``` html
<html>
  <head>
    <style>
      button {
        ...
      }
    </style>
  </head>
  <body>
    <button id="changeColor"></button>
    <script src="popup.js"></script>
  </body>
</html>
```

### popup.js
``` javascript
'use strict';

let changeColor = document.getElementById('changeColor');

chrome.storage.sync.get('color', function(data) {
  changeColor.style.backgroundColor = data.color;
  changeColor.setAttribute('value', data.color);
});

changeColor.onclick = function(element) {
  let color = element.target.value;
  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    chrome.tabs.executeScript(
        tabs[0].id,
        {code: 'document.body.style.backgroundColor = "' + color + '";'});
  });
};

```

## User Options

In `manifest.json` an option page is declared like `"options_page": "options.html"`:

### options.html
``` html
<!DOCTYPE html>
<html>
  <head>
    <style>
      button {
        ...
      }
    </style>
  </head>
  <body>
    <div id="buttonDiv">
    </div>
    <div>
      <p>Choose a different background color!</p>
    </div>
  </body>
  <script src="options.js"></script>
</html>
```

### options.js
``` javascript
'use strict';

let page = document.getElementById('buttonDiv');
const kButtonColors = ['#3aa757', '#e8453c', '#f9bb2d', '#4688f1'];
function constructOptions(kButtonColors) {
  for (let item of kButtonColors) {
    let button = document.createElement('button');
    button.style.backgroundColor = item;
    button.addEventListener('click', function() {
      chrome.storage.sync.set({color: item}, function() {
        console.log('color is ' + item);
      })
    });
    page.appendChild(button);
  }
}
constructOptions(kButtonColors);
```

