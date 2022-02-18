var gulp = require('gulp');
var markdown = require('gulp-markdown-it');

gulp.task('markdown', function() {
    return gulp.src('./md/*.md')
        .pipe(markdown())
        .pipe(gulp.dest('./html'));
});
/*
gulp.task('watch', function() {
    gulp.watch('./md/*.md', ['markdown']);
});
*/
gulp.task('default', gulp.series('markdown'/*, 'watch'*/));
