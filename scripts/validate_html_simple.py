import sys
from html.parser import HTMLParser


class Validator(HTMLParser):
    def handle_error(self, message):
        print(f"Error: {message}")
        sys.exit(1)


with open(r"src\nvision\gui\templates\index.html", encoding="utf-8") as f:
    content = f.read()

try:
    parser = Validator()
    parser.feed(content)
    parser.close()
    print("HTML seems valid (parsed without exceptions).")
except Exception as e:
    print(f"HTML validation failed: {e}")
    sys.exit(1)
