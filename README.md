# MineSAT
A python program that generates Minesweeper fields out of boolean expressions, based on Richard Kaye's article [Minesweeper is NP complete](https://link.springer.com/article/10.1007/BF03025367).

## Usage
Use this command to render `formula`:
```
python minesat.py [-s] [-d] <formula> [output.png]
```
Options:
- `-h`: Show help message
- `-s`: Only save the image (otherwise a window with the generated image will pop up)
- `-d`: Disable DPI scaling when showing the result (useful on high resolution displays with huge formulas)

Example:
```
python minesat.py -s "a+b*c" test.png
```

### Boolean Formulas
The following operators are supported:
|Name|Symbols|
|-----|------|
| NOT | `~`, `-` |
| AND | `&`, `*` |
| OR  | `\|`, `+`|
| XOR | `^`      |
| Brackets|`(`, `)`|