# Fiddler Learning Notes

## Download Links

- <https://www.telerik.com/download/fiddler/fiddler4>
- <https://telerik-fiddler.s3.amazonaws.com/fiddler/FiddlerSetup.exe>

## Fiddler Script

By default, Fiddler Script is a JScript.NET file saved under:
C:\Users\jiew\Documents\Fiddler2\Scripts\CustomRules.js

To edit the script, following ways supported:

1. Rules menu => Customize Rules... (Ctrl+R), this will bring up Fiddler ScriptEditor;
2. On default layout right pane, click FiddlerScript pane;
3. Edit C:\Users\jiew\Documents\Fiddler2\Scripts\CustomRules.js using any text editor then reload in Fiddler;

## Modify Request or Response

https://docs.telerik.com/fiddler/KnowledgeBase/FiddlerScript/ModifyRequestOrResponse

In CustomRules.js, there are 2 important functions *OnBeforeRequest* & *OnBeforeResponse*:

- static function OnBeforeRequest(oSession: Session)
- static function OnBeforeResponse(oSession: Session)

In these functions, you can write if clause on oSession object and run any scripts.

To learn the properties and methods of Session object, use intellisense or open Class View -> Session;

``` csharp
// Add a request header
oSession.oRequest["NewHeaderName"] = "New header value";

// Delete a response header
oSession.oResponse.headers.Remove("Set-Cookie");

// Change a request for one page to a different page on the same server
if (oSession.PathAndQuery=="/version1.css") {
  oSession.PathAndQuery="/version2.css";
}

// Point all requests for one server to the same port on a different server
if (oSession.HostnameIs("www.bayden.com")) {
  oSession.hostname="test.bayden.com";
}

// Point all requests for one port to a different port on a different server
if (oSession.host=="www.bayden.com:8080") {
  oSession.host="test.bayden.com:9090";
}

// Point all requests for one server to a different server, including HTTPS tunnels
// Redirect traffic, including HTTPS tunnels
if (oSession.HTTPMethodIs("CONNECT") && (oSession.PathAndQuery == "www.example.com:443")) {
    oSession.PathAndQuery = "beta.example.com:443";
}

if (oSession.HostnameIs("www.example.com")) oSession.hostname = "beta.example.com";

// Simulate the Windows HOSTS file, by pointing one Hostname to a different IP address. (Retargets without changing the request's Host header)

// All requests for subdomain.example.com should be directed to the development server at 128.123.133.123
if (oSession.HostnameIs("subdomain.example.com")){
  oSession.bypassGateway = true;                   // Prevent this request from going through an upstream proxy
  oSession["x-overrideHost"] = "128.123.133.123";  // DNS name or IP address of target server
}

// Retarget requests for a single page to a different page, potentially on a different server. (Retargets by changing the request's Host header)
if (oSession.url=="www.example.com/live.js") {
  oSession.url = "dev.example.com/workinprogress.js";
}

// Prevent upload of HTTP Cookies
oSession.oRequest.headers.Remove("Cookie");

// Decompress and unchunk a HTTP response, updating headers if needed
// Remove any compression or chunking from the response in order to make it easier to manipulate
oSession.utilDecodeResponse();

// Search and replace in HTML.
if (oSession.HostnameIs("www.bayden.com") && oSession.oResponse.headers.ExistsAndContains("Content-Type","text/html")){
  oSession.utilDecodeResponse();
  oSession.utilReplaceInResponse('<b>','<u>');
}

// Case insensitive Search of response HTML.
if (oSession.oResponse.headers.ExistsAndContains("Content-Type", "text/html") && oSession.utilFindInResponse("searchfor", false)>-1){
  oSession["ui-color"] = "red";
}

// Remove all DIV tags (and content inside the DIV tag)

// If content-type is HTML, then remove all DIV tags
if (oSession.oResponse.headers.ExistsAndContains("Content-Type", "html")){
  // Remove any compression or chunking
  oSession.utilDecodeResponse();
  var oBody = System.Text.Encoding.UTF8.GetString(oSession.responseBodyBytes);

  // Replace all instances of the DIV tag with an empty string
  var oRegEx = /<div[^>]*>(.*?)<\/div>/gi;
  oBody = oBody.replace(oRegEx, "");

  // Set the response body to the div-less string
  oSession.utilSetResponseBody(oBody);
}

// Pretend your browser is the GoogleBot webcrawler
oSession.oRequest["User-Agent"]="Googlebot/2.X (+http://www.googlebot.com/bot.html)";

// Request Hebrew content
oSession.oRequest["Accept-Language"]="he";

// Deny .CSS requests
if (oSession.uriContains(".css")) {
  oSession["ui-color"]="orange";
  oSession["ui-bold"]="true";
  oSession.oRequest.FailSession(404, "Blocked", "Fiddler blocked CSS file");
}

// Simulate HTTP Basic authentication (Requires user to enter a password before displaying web content.)
if ((oSession.HostnameIs("www.example.com")) && !oSession.oRequest.headers.Exists("Authorization")) {
  // Prevent IE's "Friendly Errors Messages" from hiding the error message by making response body longer than 512 chars.
  var oBody = "<html><body>[Fiddler] Authentication Required.<BR>".PadRight(512, ' ') + "</body></html>";
  oSession.utilSetResponseBody(oBody);
  // Build up the headers
  oSession.oResponse.headers.HTTPResponseCode = 401;
  oSession.oResponse.headers.HTTPResponseStatus = "401 Auth Required";
  oSession.oResponse["WWW-Authenticate"] = "Basic realm=\"Fiddler (just hit Ok)\"";
  oResponse.headers.Add("Content-Type", "text/html");
}

// Respond to a request with a file loaded from the \Captures\Responses folder (Can be placed in OnBeforeRequest or OnBeforeResponse function)
if (oSession.PathAndQuery=="/version1.css") {
  oSession["x-replywithfile"] ="version2.css";
}
```

## Access Fiddler UI

Use native FiddlerObject for methods like log, alert, or set StatusText

<https://docs.telerik.com/fiddler/KnowledgeBase/FiddlerScript/CustomizeMenus>

## Pause Web Sessions

``` csharp
// Pause all HTTP POSTs to allow hand-editing (the POST verb is often used for submitting forms)
if (oSession.HTTPMethodIs("POST")){
  oSession["x-breakrequest"]="breaking for POST";
}

// Pause all HTTP POSTs that contain 'thekeyword'
if (oSession.HTTPMethodIs("POST") && (oSession.utilFindInRequest("thekeyword", true) > -1)){
oSession["x-breakrequest"] = "keyword";
}

// Pause a request for an XML file to allow hand-editing
if (oSession.url.toLowerCase().indexOf(".xml")>-1){
 oSession["x-breakrequest"]="reason_XML";
}

// Pause a response containing JavaScript to allow hand-editing (in OnBeforeResponse)
if (oSession.oResponse.headers.ExistsAndContains("Content-Type", "javascript")){
  oSession["x-breakresponse"]="reason is JScript";
}
```

## Reference

- <https://www.telerik.com/blogs/understanding-fiddlerscript>
- <http://fiddlerbook.com/Fiddler/dev/ScriptSamples.asp>
