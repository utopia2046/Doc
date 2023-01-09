# UET Transition

## bat.js

### Code Location

AdsAppUI/private/UI/Tools/UET-bat-js

* bat.js:

AdsAppUI/private/UI/Tools/UET-bat-js/src/js/bat.js

* Snippet to embbed on user's site:

AdsAppUI/private/UI/Tools/UET-bat-js/src/js/snippet.js

### Development guide

#### New Test Infra

<https://microsoft.sharepoint.com/teams/BingAds/_layouts/OneNote.aspx?id=%2Fteams%2FBingAds%2FShared%20Documents%2FTeams%2FWeb%20UI%2FWiki%2FReporting%20and%20Audience&wd=target%28UET%20V2.one%7C83714845-AC45-41CC-BA83-1201D1619FF3%2FBat.js%20new%20test%20infra%7CF44835C2-73AF-40FF-82DB-2D67094B311A%2F%29>

Onenote:

<https://microsoft.sharepoint.com/teams/BingAds/Shared%20Documents/Teams/Web%20UI/Wiki/Reporting%20and%20Audience/UET%20V2.one#Bat.js%20new%20test%20infra&section-id={83714845-AC45-41CC-BA83-1201D1619FF3}&page-id={F44835C2-73AF-40FF-82DB-2D67094B311A}&end>

#### Release Process

<https://microsoft.sharepoint.com/teams/BingAds/_layouts/OneNote.aspx?id=%2Fteams%2FBingAds%2FShared%20Documents%2FTeams%2FWeb%20UI%2FWiki%2FReporting%20and%20Audience&wd=target%28UET%20V2.one%7C83714845-AC45-41CC-BA83-1201D1619FF3%2FBat.js%20release%20process%7C4063A296-D37D-4CF9-9159-619D25749A14%2F%29>

Onenote:

<https://microsoft.sharepoint.com/teams/BingAds/Shared%20Documents/Teams/Web%20UI/Wiki/Reporting%20and%20Audience/UET%20V2.one#Bat.js%20release%20process&section-id={83714845-AC45-41CC-BA83-1201D1619FF3}&page-id={4063A296-D37D-4CF9-9159-619D25749A14}&end>

## Adobe Launch

### Code Location

AdsAppUI/private/UI/Tools/UET-external-partners/Adobe-Launch-extension

### Development guide

<https://microsoft.sharepoint.com/teams/BingAdsUET/_layouts/OneNote.aspx?id=%2Fteams%2FBingAdsUET%2FShared%20Documents%2FRemarketing%20UET&wd=target%28UET.one%7C9D669A1D-B14B-4000-9A67-76A672240E73%2FAdobe%20Launch%7CBE9B70AA-685D-466E-9E64-362571525959%2F%29>

onenote:

<https://microsoft.sharepoint.com/teams/BingAdsUET/Shared%20Documents/Remarketing%20UET/UET.one#Adobe%20Launch&section-id={9D669A1D-B14B-4000-9A67-76A672240E73}&page-id={BE9B70AA-685D-466E-9E64-362571525959}&end>

### Client Credentials

API Key (Client ID)
0fb48bbc7ada4d7595783a2173e224f5

Technical account ID
BD7636215AAB2EF20A495E49@techacct.adobe.com

Technical account email
95f3679c-110d-4724-b258-70168e98f310@techacct.adobe.com

Organization ID
0F9267C25244508F0A490D4C@AdobeOrg

Client secret
ef35164e-5bdc-48b0-9080-01b95b428f74

Public keys fingerprint
00655cb26b10b7cfa940c8ce77dba30335ef4629

### Demo page

<https://bingadsuet.azurewebsites.net/BingAdsUETExtensionForAdobeLaunch_DemoNew.html>

## Google GTM

### Code location

Latest code is in Github repo gtm-vender-templates/microsoft, branch `baut2`:

https://github.com/gtm-vendor-templates/microsoft/tree/baut2

> In master branch, it is the previous version Google engineer created.

Spec:

AdsAppUI/private/UI/Tools/UET-external-partners/GTM-template/spec/md/BingAds UET Template for Google Tag Manager Design.md

Preview:

https://msasg.visualstudio.com/Bing_Ads/_git/AdsAppUI?_a=preview&path=%2Fprivate%2FUI%2FTools%2FUET-external-partners%2FGTM-template%2Fspec%2Fmd%2FBingAds%20UET%20Template%20for%20Google%20Tag%20Manager%20Design.md&version=GBmaster

### Accounts

Google Tag Manager:

<https://tagmanager.google.com>

bingadsuetv2@gmail.com
Microsoft~2

GTM developer guide:

<https://developers.google.com/tag-manager/quickstart>
<https://developers.google.com/tag-manager/devguide#events>
<https://support.google.com/tagmanager/answer/7683362>
<https://support.google.com/tagmanager/answer/6107056>

GTM vendor template development documents:

<https://developers.google.com/tag-manager/templates>
<https://developers.google.com/tag-manager/templates/api>
<https://developers.google.com/tag-manager/templates/sandboxed-javascript>

These document are beta so you need to sign in GTM using bingadsuetv2 account to see them.

Github repo:

<https://github.com/gtm-vendor-templates/microsoft>

bing-ads-uet
BingAds123

The primary and backup email of Github account could be changed so that we won't need to ask Google support to new account access.

Enterprise Github account requires 2-factor authentication, so you need to install one of following password app:

* 1Password
* Authy
* LastPass Authenticator

Please see Github help for detail steps:

<https://help.github.com/en/articles/configuring-two-factor-authentication>

### TODO

Current we're working with Google Support Alex Bogart for possible solutions to support Sandboxed JS. Current blocking is that Sandboxed JS doesn't support new object, so there are (at least) 2 options:

1. We create a snippet to create UET object and update build/deploy script to deploy a GTM specific version bat.js;
2. Continue use old JS template, need Google confirm of how long the old JS template will be supported;

All the discussion with Alex Bogart could be found at PR:

https://github.com/gtm-vendor-templates/microsoft/pull/1

## Dashboard

### Current dashboards

* https://pulseweb.microsoft.com/report/uet
* https://pulseweb.microsoft.com/report/remarket
* https://pulseweb.microsoft.com/report/remarketinglist_for_dev
* https://pulseweb.microsoft.com/report/customaudience
* https://pulseweb.microsoft.com/report/inmarketaudience
* https://pulseweb.microsoft.com/report/productaudience
* https://pulseweb.microsoft.com/report/itp
* https://pulseweb.microsoft.com/report/MsClkIdFunnel

### Aggregation Scripts

AudiencesAssociationNewSegmentReport_CosmosScript

https://avocado/a/#/tasks/221296

AudiencesAssociationNewSegmentReportOnAdinsightsData_CosmosScript

https://avocado/a/#/tasks/224491

CustomAndInMarketAudience_DataPrepare

https://avocado/a/#/tasks/227431

AudiencesAssociation_HandleWeeklyData_CosmosScript

https://avocado/a/#/tasks/227237

AudiencesAssociationDataPreparationOnAdinsightsData_CosmosScript

https://avocado/a/#/tasks/312429

RemarketingUETDataPreparation_ConversionsClicksGoalsTags_CosmosScript

https://avocado/a/#/tasks/312780

CampaignLevelAudienceAssociation_UsageReport

https://avocado/a/#/tasks/312300

BingAds_Audience_CampaignLevelAssociation_Report

https://avocado/a/#/tasks/312430

UETReport - Cosmos Script

https://avocado/a/#/tasks/148426

UETReport_NewSegment_Forrest - CosmosScript

https://avocado/a/#/tasks/226280

UET_ITP_NonJSConversions

https://avocado/a/#/tasks/287276

UET_ITP_GenerateAutoTaggingFunnel

https://avocado/a/#/tasks/290070

### DB upload jobs

Under all bingadswebui jobs, search "DBLoad"

https://avocado/a/#/search/basic/all/bingadswebui

### TODO

Currently there is a shared stream design draft and the scripts are already hosted on Avocado:

https://msasg.visualstudio.com/Bing_Ads/_git/AdsAppUI/pullrequest/1011564?_a=overview

AudiencesAssociationDataPreparationOnAdinsightsData_CosmosScript

https://avocado/a/#/tasks/312429

RemarketingUETDataPreparation_ConversionsClicksGoalsTags_CosmosScript

https://avocado/a/#/tasks/312780

There are still some incoming requests for adding more properties on these shared streams, please see the mail threads for open discussion.
