/* GLOBAL SETTINGS */

# Suppress warnings
SET @@global.sql_mode = ''; # To allow for insertion of dataset from pandas
-- SELECT @@global.sql_mode;
SET SQL_SAFE_UPDATES = 0; # To allow for updates using non-key columns in the WHERE clause

/*
# Defaults
SET @@global.sql_mode = 'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION';
SET SQL_SAFE_UPDATES = 1;
*/