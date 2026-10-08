# Qualitative samples (Section 4.6)

Five correct and five incorrect dev predictions from beam search (beam size 4). SQL uses the real column names.

## Correct

### Correct 1

**Question:** What position does the player who played for butler cc (ks) play?

**Gold SQL:** `SELECT Position FROM table WHERE School/Club Team = 'Butler CC (KS)'`

**Predicted SQL:** `SELECT Position FROM table WHERE School/Club Team = 'butler cc (ks)'`

### Correct 2

**Question:** How many schools did player number 3 play at?

**Gold SQL:** `SELECT COUNT School/Club Team FROM table WHERE No. = '3'`

**Predicted SQL:** `SELECT COUNT School/Club Team FROM table WHERE No. = '3'`

### Correct 3

**Question:** What school did player number 21 play for?

**Gold SQL:** `SELECT School/Club Team FROM table WHERE No. = '21'`

**Predicted SQL:** `SELECT School/Club Team FROM table WHERE No. = '21'`

### Correct 4

**Question:** Who is the player that wears number 42?

**Gold SQL:** `SELECT Player FROM table WHERE No. = '42'`

**Predicted SQL:** `SELECT Player FROM table WHERE No. = '42'`

### Correct 5

**Question:** What player played guard for toronto in 1996-97?

**Gold SQL:** `SELECT Player FROM table WHERE Position = 'Guard' AND Years in Toronto = '1996-97'`

**Predicted SQL:** `SELECT Player FROM table WHERE Position = 'guard' AND Years in Toronto = '1996-97'`

## Wrong

### Wrong 1: Wrong WHERE value

**Question:** Who are all of the players on the Westchester High School club team?

**Gold SQL:** `SELECT Player FROM table WHERE School/Club Team = 'Westchester High School'`

**Predicted SQL:** `SELECT Player FROM table WHERE School/Club Team = 'westchester high'`

**Failure:** Wrong WHERE value.

### Wrong 2: Wrong WHERE column

**Question:** When did Jacques Chirac stop being a G8 leader?

**Gold SQL:** `SELECT Ended time as senior G8 leader FROM table WHERE Person = 'Jacques Chirac'`

**Predicted SQL:** `SELECT Ended time as senior G8 leader FROM table WHERE Entered office as Head of State or Government = 'jacques chirac'`

**Failure:** Wrong WHERE column.

### Wrong 3: Missing WHERE condition

**Question:** What position is number 35 whose height is 6-6?

**Gold SQL:** `SELECT Position FROM table WHERE Height in Ft. = '6-6' AND No.(s) = '35'`

**Predicted SQL:** `SELECT Position FROM table WHERE Height in Ft. = '6-6'`

**Failure:** Missing WHERE condition.

### Wrong 4: Wrong WHERE conditions and values

**Question:** what is the bristol & n. som where the somerset is ashcott and shapwick?

**Gold SQL:** `SELECT Bristol & N. Som FROM table WHERE Somerset = 'Ashcott and Shapwick'`

**Predicted SQL:** `SELECT Bristol & N. Som FROM table WHERE Glos & Wilts = 'ashcott' AND Bristol & Somerset = 'shapwick'`

**Failure:** Wrong WHERE conditions and values.

### Wrong 5: Parse failure

**Question:** Which launch date involved the Driade?

**Gold SQL:** `SELECT Launched FROM table WHERE Name = 'Driade'`

**Predicted SQL:** `Parse failure: output could not be parsed as a WikiSQL query.`

**Failure:** Parse failure.
