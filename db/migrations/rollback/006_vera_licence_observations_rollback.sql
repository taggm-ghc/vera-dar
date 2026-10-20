-- 006_vera_licence_observations ROLLBACK -- DESTRUCTIVE: drops vera_vjay.licence_observations and every observation in it.
-- Requires R1's separate, same-turn approval and an admin account. Run only after a pg_dump of the table.
-- Non-destructive rollback is: set the config flag off, then REVOKE INSERT from the writer roles.

DROP TABLE IF EXISTS vera_vjay.licence_observations;
DROP FUNCTION IF EXISTS vera_vjay.licence_observations_forbid_mutation();
