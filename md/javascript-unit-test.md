# CommonJS style module unit testing

## Sample module

``` javascript
// team.js
var User = require('./user');

function getTeam(teamId) {
    return User.find({teamId: teamId});
}

module.exports.getTeam = getTeam;
```

## Sample test

``` javascript
// team.spec.js
var Team = require('./team');
var User = require('./user');

describe('Team test', function() {
    it('getTeam test', function() {
        var users = [{id:1, id:2}];

        this.sandbox.stub(User, 'find', function() {
            return Promise.resolve(users);
        });

        var team = yield team.getTeam();

        expect(team).to.eql(users);
    });
});
```
