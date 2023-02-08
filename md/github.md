# Github Tricks

## Clone github repro in SSH

### 1. Generate SSH public/private key pairs.

Open git command console, under home path ~

``` console
ssh-keygen -t rsa -C "utopia2046@hotmail.com"
```

The pass phrase and filename input could be ignored. That would use default file names and with empty pass phrase

Check the generated id_rsa (private key) and id_rsa.pub (public key) files under .ssh folder. Copy content from id_rsa.pub.

Test the generated key pair using:

``` console
ssh -T git@github.com
```

When seeing below warning, input yes to continue.

``` console
This key is not known by any other names
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
```

### 2. Add SSH key on Github website.

Sign in Github website and go to https://github.com/settings/keys page. Click New SSH Key, give it a name and paste the public key content.

### 3. Clone repo to local.

Go to your repro, copy clone string from Code menue SSH tab like below, and run in Git console:

``` console
git clone git@github.com:utopia2046/Doc.git
```

Install required libs like:

``` console
npm install --global gulp
npm install --global yarn
yarn install
```
