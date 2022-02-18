# Site Template JSON

Site template JSON follows our core-interface.ts hierarchy of site -> section -> group -> component.

This example site template contains 2 pages:

* Overview page: with an external image, business name, description, and a link to contact page.
* Contact page: with address, phone number, e-mail, and a link to overview page.

The dataSource member stores properties that we expose to user in basic site setting editor. And when user update the property values, they will be saved in the corresponding components content.

``` javascript
{
  id: '',
  title: '',
  dataSource: [{
    varName: 'BUSINESS_NAME',
    label: 'Business Name',
    type: 'string',
    value: '',
  }, {
    varName: 'DESCRIPTION',
    label: 'Description',
    type: 'string',
    value: '',
  }, {
    varName: 'ADDRESS',
    label: 'Address',
    type: 'string',
    value: '',
  }, {
    varName: 'PHONE',
    label: 'Phone number',
    type: 'string',
    value: '',
  }, {
    varName: 'EMAIL',
    label: 'Email',
    type: 'string',
    value: '',
  }],
  mediaResources : [{
    type: 'image',
    id: '7a84ac8a-a9fd-4121-8da2-b505cee72554',
    url: 'https://docs.microsoft.com/en-us/microsoft-365/media/contoso-overview/contoso-overview-fig1.png'
  }],
  pages: [{
    id: 'page-ofqwnb',
    name: 'Overview',
    sections: [{
      id: 'sec-m6qj7j',
      groups: [{
        id: 'grp-6v6d59',
        components: [{
          id: 'cmp-9g92h8',
          type: 'image',
          style: 'width:948px;height:462px;',
          resouceId: '7a84ac8a-a9fd-4121-8da2-b505cee72554',
        }],
      }],
    }, {
      id: 'sec-q4y8ml',
      groups: [{
        id: 'grp-saxkxe',
        components: [{
          id: 'cmp-pkbarg',
          type: 'text',
          style: 'text-align:center;font-size:48px;font-weight:600;line-height:72px;color:#605e5c;margin-top:20px;margin-bottom:40px;',
          content: '{{BUSINESS_NAME}}',
        }],
      }],
    }{
      id: 'sec-1jl2wz',
      groups: [{
        id: 'grp-jjqz8y',
        components: [{
          id: 'cmp-iphf8p',
          type: 'text',
          style: 'text-align:center;font-size:16px;line-height:24px;color:#323130;margin-left:20%;margin-right:20%;',
          content: '{{DESCRIPTION}}',
        }],
      }],
    }{
      id: 'sec-b9xyqw',
      groups: [{
        id: 'grp-9uiute',
        components: [{
          id: 'cmp-5hwsy1',
          type: 'link',
          style: 'display:block;text-align:center;font-size:18px;font-weight:600;line-height:72px;margin-top:20px;margin-bottom:20px;color:#0078d4',
          target: '#page-x468ik',
          content: 'CONTACT US',
        }],
      }],
    }],
  }, {
    id: 'page-x468ik',
    name: 'Contact',
    sections: [{
      id: 'sec-kyl6tl',
      style: 'margin-top:150px;margin-bottom:150px',
      groups: [{
        id: 'grp-msk3ql',
        components: [{
          id: 'cmp-mdjkzn',
          type: 'text',
          style: 'text-align:center;font-size:24px;font-weight:600;line-height:32px;color:#605e5c;margin-top:20px;',
          content: 'Address: {{ADDRESS}}',
        }, {
          id: 'cmp-zait2s',
          type: 'text',
          style: 'text-align:center;font-size:24px;font-weight:600;line-height:32px;color:#605e5c;margin-top:20px;',
          content: 'Phone Number: {{PHONE}}',
        }, {
          id: 'cmp-mdjkzn',
          type: 'text',
          style: 'text-align:center;font-size:24px;font-weight:600;line-height:32px;color:#605e5c;margin-top:20px;',
          content: 'E-Mail: {{EMAIL}}',
        }],
      }],
    }, {
      id: 'sec-kfwq04',
      groups: [{
        id: 'grp-z9rc92',
        components: [{
          id: 'cmp-2nuxuw',
          type: 'link',
          style: 'display:block;text-align:center;font-size:18px;font-weight:600;line-height:72px;margin-top:20px;margin-bottom:20px;color:#0078d4',
          target: '#page-ofqwnb',
          content: 'BACK TO TOP',
        }],
      }],
    }],
  }],
}
```

# Saved JSON

After user update site property values in editor and hit Save button, the site data will be saved like following:

``` javascript
{
  id: '186478652873',
  title: 'Contoso Homepage',
  mediaResources : [{
    type: 'image',
    id: '7a84ac8a-a9fd-4121-8da2-b505cee72554',
    url: 'https://docs.microsoft.com/en-us/microsoft-365/media/contoso-overview/contoso-overview-fig1.png'
  }],
  pages: [{
    id: 'page-ofqwnb',
    name: 'Overview',
    sections: [{
      id: 'sec-m6qj7j',
      groups: [{
        id: 'grp-6v6d59',
        components: [{
          id: 'cmp-9g92h8',
          type: 'image',
          style: 'width:948px;height:462px;',
          resouceId: '7a84ac8a-a9fd-4121-8da2-b505cee72554',
        }],
      }],
    }, {
      id: 'sec-q4y8ml',
      groups: [{
        id: 'grp-saxkxe',
        components: [{
          id: 'cmp-pkbarg',
          type: 'text',
          style: 'text-align:center;font-size:48px;font-weight:600;line-height:72px;color:#605e5c;margin-top:20px;margin-bottom:40px;',
          content: 'CONTOSO',
        }],
      }],
    }{
      id: 'sec-1jl2wz',
      groups: [{
        id: 'grp-jjqz8y',
        components: [{
          id: 'cmp-iphf8p',
          type: 'text',
          style: 'text-align:center;font-size:16px;line-height:24px;color:#323130;margin-left:20%;margin-right:20%;',
          content: 'The Contoso Corporation is a fictional but representative global manufacturing conglomerate with its headquarters in Paris, France. Contoso has deployed Microsoft 365 Enterprise and addressed major design decisions and implementation details for networking, identity, Windows 10 Enterprise, Microsoft 365 Apps for enterprise, mobile device management, information protection, and security.',
        }],
      }],
    }{
      id: 'sec-b9xyqw',
      groups: [{
        id: 'grp-9uiute',
        components: [{
          id: 'cmp-5hwsy1',
          type: 'link',
          style: 'display:block;text-align:center;font-size:18px;font-weight:600;line-height:72px;margin-top:20px;margin-bottom:20px;color:#0078d4',
          target: '#page-x468ik',
          content: 'CONTACT US',
        }],
      }],
    }],
  }, {
    id: 'page-x468ik',
    name: 'Contact',
    sections: [{
      id: 'sec-kyl6tl',
      style: 'margin-top:150px;margin-bottom:150px',
      groups: [{
        id: 'grp-msk3ql',
        components: [{
          id: 'cmp-mdjkzn',
          type: 'text',
          style: 'text-align:center;font-size:24px;font-weight:600;line-height:32px;color:#605e5c;margin-top:20px;',
          content: 'Address: Paris',
        }, {
          id: 'cmp-zait2s',
          type: 'text',
          style: 'text-align:center;font-size:24px;font-weight:600;line-height:32px;color:#605e5c;margin-top:20px;',
          content: 'Phone Number: 123-456-7890',
        }, {
          id: 'cmp-mdjkzn',
          type: 'text',
          style: 'text-align:center;font-size:24px;font-weight:600;line-height:32px;color:#605e5c;margin-top:20px;',
          content: 'E-Mail: admin@contoso.com',
        }],
      }],
    }, {
      id: 'sec-kfwq04',
      groups: [{
        id: 'grp-z9rc92',
        components: [{
          id: 'cmp-2nuxuw',
          type: 'link',
          style: 'display:block;text-align:center;font-size:18px;font-weight:600;line-height:72px;margin-top:20px;margin-bottom:20px;color:#0078d4',
          target: '#page-ofqwnb',
          content: 'BACK TO TOP',
        }],
      }],
    }],
  }],
}
```

# Rendered HTML

Rendering engine will read site json and generate HTML like following.

``` html
<!DOCTYPE html>
<html lang="en">

<head>
  <title>Contoso Homepage</title>
  <meta charset="utf-8" />
  <!-- common styles for built-in classes
  <style id="built-in-styles" type="text/css">
  .page {
    font-family: Segoe UI, SegoeUI, Helvetica Neue, Helvetica, Arial, sans-serif;
    margin: 32px 20px;
    min-height: 400px;
  }
  .section {
    width: 100%;
    margin-top: 10px;
    margin-bottom: 10px;
  }
  .group {
    display: block;
  }
  .component: {
    display: block;
  }
  </style>
  -->

</head>

<body>

  <!-- render engine generated html code starts from here -->
  <div id="page-ofqwnb" class="page" data-name="Overview">
    <div id="sec-m6qj7j" class="section">
      <div id="grp-6v6d59" class="group">
        <img id="cmp-9g92h8" class="component image" style="width:948px;height:462px;" src="https://docs.microsoft.com/en-us/microsoft-365/media/contoso-overview/contoso-overview-fig1.png" />
      </div>
    </div>
    <div id="sec-q4y8ml" class="section">
      <div id="grp-saxkxe" class="group">
        <div id="cmp-pkbarg" class="component text" style="text-align:center;font-size:48px;font-weight:600;line-height:72px;color:#605e5c;margin-top:20px;margin-bottom:40px;">
          CONTOSO
        </div>
      </div>
    </div>
    <div id="sec-1jl2wz" class="section">
      <div id="grp-jjqz8y" class="group">
        <div id="cmp-iphf8p" class="component text" style="text-align:center;font-size:16px;line-height:24px;color:#323130;margin-left:20%;margin-right:20%;">
          The Contoso Corporation is a fictional but representative global manufacturing conglomerate with its headquarters in Paris, France. Contoso has deployed Microsoft 365 Enterprise and addressed major design decisions and implementation details for networking, identity, Windows 10 Enterprise, Microsoft 365 Apps for enterprise, mobile device management, information protection, and security.
        </div>
      </div>
    </div>
    <div id="sec-b9xyqw" class="section">
      <div id="grp-9uiute" class="group">
        <a id="cmp-5hwsy1" class="component link" style="display:block;text-align:center;font-size:18px;font-weight:600;line-height:72px;margin-top:20px;margin-bottom:20px;color:#0078d4" href="#page-x468ik">
          CONTACT US
        </a>
      </div>
    </div>

  </div>

  <div id="page-x468ik" class="page" data-name="Contact">
    <div id="sec-kyl6tl" class="section" style="margin-top:150px;margin-bottom:150px">
      <div id="grp-msk3ql" class="group">
        <div id="cmp-mdjkzn" class="component text" style="text-align:center;font-size:24px;font-weight:600;line-height:32px;color:#605e5c;margin-top:20px;">
          Address: Paris
        </div>
        <div id="cmp-zait2s" class="component text" style="text-align:center;font-size:24px;font-weight:600;line-height:32px;color:#605e5c;margin-top:20px;">
          Phone Number: 123-456-7890
        </div>
        <div id="cmp-vmvfyp" class="component text" style="text-align:center;font-size:24px;font-weight:600;line-height:32px;color:#605e5c;margin-top:20px;">
          E-Mail: admin@contoso.com
        </div>
      </div>
    </div>
    <div id="sec-kfwq04" class="section">
      <div id="grp-z9rc92" class="group">
        <a id="cmp-2nuxuw" class="component link" style="display:block;text-align:center;font-size:18px;font-weight:600;line-height:72px;margin-top:20px;margin-bottom:20px;color:#0078d4" href="#page-ofqwnb">
          BACK TO TOP
        </a>
      </div>
    </div>
  </div>
  <!-- render engine generated html code ends at here -->

</body>

</html>
```

