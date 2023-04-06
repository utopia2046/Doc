# Markdown Typography

## Work in Visual Studio Code

- Press `Control-K V` to open preview window.
- Press `Control-Shift P` and enter markdown for more markdown commands.
- You can disable scroll synchronization using the `markdown.preview.scrollPreviewWithEditor` and `markdown.preview.scrollEditorWithPreview` settings.
- Press `Ctrl+Space` (Trigger Suggest) and you get a context specific list of suggestions.

## Compile

Install markdown-it:

`npm install -g markdown-it`

Run markdown-it on md files:

`markdown-it md/markdown.md -o html/markdown.html`

Now when you edit any .md file, the gulp task will run automatically and generate output .html files.

Reference: <https://code.visualstudio.com/docs/languages/markdown#_compiling-markdown-into-html>

## Headings

Headings from `h1` through `h6` are constructed with a `#` for each level:

``` markdown
# h1 Heading
## h2 Heading
### h3 Heading
#### h4 Heading
##### h5 Heading
###### h6 Heading
```

Alternatively, for H1 and H2, an underline-ish style:

``` markdown
Alt-H1
======

Alt-H2
------
```

## Horizontal lines

The HTML `<hr>` element is for creating a "thematic break" between paragraph-level elements. In markdown, you can create a `<hr>` with any of the following:

- `___`: three consecutive underscores
- `---`: three consecutive dashes
- `***`: three consecutive asterisks

## Emphasis

<!-- markdownlint-disable -->
- *italic text with stars*
- _italic text with underscores_
- **bold text with 2 stars**
- __bold text with 2 underscores__
- ***bold italic text with 3 stars
- ~~strike through text with tilde~~
<!-- markdownlint-enable -->

## Blockquotes

For quoting blocks of content from another source within your document.
Add `>` before any text you want to quote.

> Lorem ipsum dolor sit amet, consectetur adipiscing elit.
> Integer posuere erat a ante.

Blockquotes can also be nested:

> Donec massa lacus, ultricies a ullamcorper in, fermentum sed augue.
Nunc augue augue, aliquam non hendrerit ac, commodo vel nisi.
>> Sed adipiscing elit vitae augue consectetur a gravida nunc vehicula. Donec auctor
odio non est accumsan facilisis.
>>> Donec massa lacus, ultricies a ullamcorper in, fermentum sed augue.
Nunc augue augue, aliquam non hendrerit ac, commodo vel nisi.

## Lists

### Unordered lists

A list of items in which the order of the items does not explicitly matter.

You may use any of the following symbols to denote bullets for each list item:

```markdown
* valid bullet
- valid bullet
+ valid bullet
```

### Ordered lists

A list of items in which the order of items does explicitly matter.

1. Lorem ipsum dolor sit amet
2. Consectetur adipiscing elit
3. Integer molestie lorem at massa

### Nested lists

1. First ordered list item
2. Another item
    - Unordered sub-list.
        - sub unordered list.
3. Actual numbers don't matter, just that it's a number
    1. Ordered sub-list
4. And another item.

   You can have properly indented paragraphs within list items.
   Notice the blank line above, and the leading spaces
   (at least one, but we'll use three here to also align the raw Markdown).

## Code

### Inline code

Wrap inline snippets of code with a pair of grave accents `` ` ``.

For example, `<section></section>` should be wrapped as "inline".

### Indented code

Or indent several lines of code by at least four spaces, as in:

``` js
    // Some comments
    line 1 of code
    line 2 of code
    line 3 of code
```

### Block code "fences"

Use "fences" (3 grave accents) ```` ``` ```` to block in multiple lines of code.

``` html
<pre>
  <p>Sample text here...</p>
</pre>
```

### Syntax highlighting

Add the file extension of the language you want to use directly after the first
code "fence", ` ``` js `, and syntax highlighting will automatically be applied
in the rendered HTML. For example, to apply syntax highlighting to JavaScript code:

``` javascript
grunt.initConfig({
  assemble: {
    options: {
      data: 'src/data/*.{json,yml}',
    },
    pages: {
      files: {
        './': ['src/templates/pages/index.hbs']
      }
    }
  }
};
```

Common supported languages:

- cpp, c
- c#, c-sharp, csharp
- css
- java
- js, jscript, javascript
- text, plain
- py, python
- sql
- xml, hthml, xslt, html
- r, s, splus

## Tables

Tables are created by adding pipes as dividers between each cell,
and by adding a line of dashes (also separated by bars) beneath the header.
Install markdown prettifier extension to auto format the table using `Ctrl+Alt+M`

| Option | Description                                 |
|--------|---------------------------------------------|
| data   | path to data files to supply the data.      |
| engine | engine to be used for processing templates. |
| ext    | extension to be used for dest files.        |

Add `:` to the right end of column to enable align-right.

| Option |                                 Description |
|-------:|--------------------------------------------:|
|   data |      path to data files to supply the data. |
| engine | engine to be used for processing templates. |
|    ext |        extension to be used for dest files. |

## Links

### Basic link

[Assemble](http://assemble.io)

### Add a title

[Upstage](https://github.com/upstage/ "Visit Upstage tooltip!")

### Named Anchors

```markdown
# Table of Contents
  * [Chapter 1](#chapter-1)
  * [Chapter 2](#chapter-2)
  * [Chapter 3](#chapter-3)

## Chapter 1 <a id="chapter-1"></a>
Content for chapter one.

## Chapter 2 <a id="chapter-2"></a>
Content for chapter one.

## Chapter 3 <a id="chapter-3"></a>
Content for chapter one.
```

## Images

Images have a similar syntax to links but include a preceding exclamation point.

![Minion Alt Text](../images/minion.jpg)

or

![Alt text](http://octodex.github.com/images/stormtroopocat.jpg "The Stormtroopocat")

Like links, Images also have a footnote style syntax

![Alt text][id]

With a reference later in the document defining the URL location:

[id]: http://octodex.github.com/images/dojocat.jpg  "The Dojocat"
