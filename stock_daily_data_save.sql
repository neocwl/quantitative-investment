CREATE TABLE `stock_BJ`(
    `id` bigint(18) NOT NULL AUTO_INCREMENT,
    `ts_code` VARCHAR(30),
    `trade_date` VARCHAR(30),
    `open` DOUBLE,
    `high` DOUBLE,
    `low` DOUBLE,
    `close` DOUBLE,
    `pre_close` DOUBLE,
    `change` DOUBLE,
    `pct_chg` DOUBLE,
    `vol` DOUBLE,
    `amount` DOUBLE,
    primary key (`id`),
    INDEX  (`trade_date`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;