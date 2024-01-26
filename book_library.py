books = []


def add_book(book_id, title, author):
    books.append({"id": book_id, "title": title, "author": author})


def borrow_book(book_id, user_id):
    for book in books:
        if book["id"] == book_id:
            book["borrowed_by"] = user_id
            return True
    return False


def return_book(book_id):
    for book in books:
        if book["id"] == book_id:
            book["borrowed_by"] = None
            return True
    return False


class Book:
    def __init__(self, id, title, author):
        self.id = id
        self.title = title
        self.author = author
        self.borrowed_by = None

    def borrow(self, user_id):
        if self.borrowed_by is None:
            self.borrowed_by = user_id
            return True
        return False

    def return_book(self):
        self.borrowed_by = None


class Library:
    def __init__(self):
        self.books = []

    def add_book(self, book):
        self.books.append(book)

    def borrow_book(self, book_id, user_id):
        for book in self.books:
            if book.id == book_id:
                return book.borrow(user_id)
        return False

    def return_book(self, book_id):
        for book in self.books:
            if book.id == book_id:
                book.return_book()
                return True
        return False


if __name__ == "__main__":
    # Creating books
    add_book(1, "Book 1", "Author 1")
    add_book(2, "Book 2", "Author 2")

    # Borrowing and returning books
    borrow_book(1, "User1")
    return_book(1)

    # Creating books and a library
    library = Library()
    library.add_book(Book(1, "Book 1", "Author 1"))
    library.add_book(Book(2, "Book 2", "Author 2"))

    # Borrowing and returning books
    library.borrow_book(1, "User1")
    library.return_book(1)

    print("Done!")
